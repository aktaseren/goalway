#!/usr/bin/env bash
#
# One-command production deploy of the Goalway.
#
#   ./deploy/deploy.sh                     build image + deploy everything
#   ./deploy/deploy.sh --skip-build        reuse the existing goalway:v1.0.0 image
#   ./deploy/deploy.sh --with-keycloak     also deploy a demo Keycloak (deploy/keycloak)
#   ./deploy/deploy.sh --project NAME      deploy into project NAME (default: goalway)
#
# Requires: oc, logged in to the target cluster.
set -euo pipefail

DEFAULT_PROJECT="goalway"
PROJECT="$DEFAULT_PROJECT"
SKIP_BUILD=0
WITH_KEYCLOAK=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------- flags ---
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build)    SKIP_BUILD=1 ;;
    --with-keycloak) WITH_KEYCLOAK=1 ;;
    --project)       shift; PROJECT="${1:?--project needs a value}" ;;
    --project=*)     PROJECT="${1#*=}" ;;
    -h|--help)       usage; exit 0 ;;
    *)               echo "error: unknown flag: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

# ------------------------------------------------------------ oc checks ---
if ! command -v oc >/dev/null 2>&1; then
  echo "error: 'oc' not found -- install the OpenShift CLI first" >&2
  exit 1
fi

if ! oc whoami >/dev/null 2>&1; then
  echo "error: not logged in -- run 'oc login <cluster-url>' first" >&2
  exit 1
fi

echo "==> Logged in as $(oc whoami) on $(oc whoami --show-server)"

# --------------------------------------------------------- project check ---
# The manifests are namespace-scoped; the project must exist first.
if oc get project "$PROJECT" >/dev/null 2>&1; then
  echo "==> Using existing project '$PROJECT'"
else
  echo "==> Project '$PROJECT' does not exist"
  read -r -p "    Create it now? [Y/n] " answer
  case "$answer" in
    n*|N*) echo "aborted -- create it with: oc new-project $PROJECT" >&2; exit 1 ;;
    *)     oc new-project "$PROJECT" >/dev/null ;;
  esac
fi
oc project "$PROJECT" >/dev/null

# ------------------------------------------------------ apply manifests ---
# Numeric prefixes ensure ordering (configmap/pvc/redis before the
# deployment). secret.example.yaml is only applied on first run.
echo "==> Applying deploy/openshift manifests in project '$PROJECT'"
for manifest in "$SCRIPT_DIR"/openshift/[0-9]*.yaml; do
  echo "    oc apply -f $(basename "$manifest")"
  oc apply -f "$manifest" >/dev/null
done

if ! oc get secret goalway-secret >/dev/null 2>&1; then
  echo "==> Creating goalway-secret from secret.example.yaml"
  echo "    NOTE: placeholder values -- replace with real credentials:"
  echo "          oc apply -f deploy/openshift/secret.example.yaml  # after editing"
  oc apply -f "$SCRIPT_DIR/openshift/secret.example.yaml" >/dev/null
fi

# The deployment manifests pin images to the default project's
# ImageStreams; repoint them when deploying elsewhere.
if [ "$PROJECT" != "$DEFAULT_PROJECT" ]; then
  echo "==> Repointing images to project '$PROJECT'"
  oc set image deployment/goalway \
    "gateway=image-registry.openshift-image-registry.svc:5000/$PROJECT/goalway:v1.0.0" \
    >/dev/null
  oc set image deployment/echo-service \
    "echo=image-registry.openshift-image-registry.svc:5000/$PROJECT/echo-service:latest" \
    >/dev/null
fi

# --------------------------------------------------------------- build ---
if [ "$SKIP_BUILD" -eq 1 ]; then
  echo "==> Skipping build (--skip-build), reusing goalway:v1.0.0"
else
  echo "==> Starting echo backend build"
  oc start-build echo-service --from-dir="$REPO_DIR" --follow

  echo "==> Starting gateway image build (binary build from $(basename "$REPO_DIR"))"
  oc start-build goalway --from-dir="$REPO_DIR" --follow
fi

# ------------------------------------------------------------ rollout ---
echo "==> Waiting for echo backend rollout"
oc rollout status deployment/echo-service --timeout=300s

echo "==> Waiting for deployment rollout"
oc rollout status deployment/goalway --timeout=300s

# ------------------------------------------------------ optional extras ---
if [ "$WITH_KEYCLOAK" -eq 1 ]; then
  echo "==> Deploying Keycloak (deploy/keycloak/keycloak.yaml)"
  oc apply -f "$SCRIPT_DIR/keycloak/keycloak.yaml" >/dev/null
  oc rollout status deployment/keycloak --timeout=300s
  KC_HOST="$(oc get route keycloak -o jsonpath='{.spec.host}')"
  echo ""
  echo "    Keycloak:   https://$KC_HOST  (admin/admin -- change me)"
  echo "    Next steps: deploy/keycloak/README.md (realm 'gateway',"
  echo "                client 'goalway', then set OIDC_* in the secret)"
fi

# ---------------------------------------------------------------- done ---
ROUTE_HOST="$(oc get route goalway -o jsonpath='{.spec.host}')"
GATEWAY_URL="http://$ROUTE_HOST"

echo ""
echo "======================================================================="
echo " Goalway deployed in project '$PROJECT'"
echo "======================================================================="
echo ""
echo "  URL: $GATEWAY_URL"
echo "  Demo console (UI): $GATEWAY_URL/ui"
echo ""
echo "  Health check:"
echo "    curl -s $GATEWAY_URL/health | python3 -m json.tool"
echo ""
echo "  Chat completion (echo backend, no GPU needed):"
echo "    curl -s -X POST $GATEWAY_URL/v1/chat/completions \\"
echo "      -H 'X-API-Key: redhat-demo-key' -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"echo\", \"messages\": [{\"role\": \"user\", \"content\": \"hello gateway\"}]}'"
echo ""
echo "  RAG search:"
echo "    curl -s -X POST $GATEWAY_URL/v1/rag/search \\"
echo "      -H 'X-API-Key: redhat-demo-key' -H 'Content-Type: application/json' \\"
echo "      -d '{\"query\": \"OpenShift AI model serving\", \"include_stats\": true}'"
echo ""
echo "  MCP initialize:"
echo "    curl -s -X POST $GATEWAY_URL/mcp \\"
echo "      -H 'X-API-Key: redhat-demo-key' -H 'Content-Type: application/json' \\"
echo "      -d '{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"initialize\", \"params\": {}}'"
echo ""
echo "  Full end-to-end demo:"
echo "    ./examples/demo.sh $GATEWAY_URL"
echo ""
