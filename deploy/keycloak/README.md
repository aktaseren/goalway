# Keycloak for the gateway (optional OIDC provider)

A single-pod, ephemeral Keycloak 26.0 for trying out the gateway's
OIDC auth. Deployed with:

```bash
./deploy/deploy.sh --with-keycloak
# or manually:
oc apply -f deploy/keycloak/keycloak.yaml
```

⚠️ Everything here is demo-grade: `admin/admin` bootstrap credentials,
no persistent storage. Do not use for anything real.

## 1. Create the realm and client

Log in at the Keycloak route (`oc get route keycloak`) as `admin` /
`admin`, then:

1. **Realm**: create a realm named `gateway`.
2. **Client**: in that realm, create a client with
   - Client ID: `goalway`
   - Client authentication: **on** (confidential) if you want a
     client secret for the password grant, or off for a public demo
     client
   - Valid redirect URIs: whatever you need (not used by the gateway
     itself, the gateway only validates tokens)
3. **Audience**: add a client scope (or dedicated mapper) so tokens
   carry `aud: ["goalway"]` — e.g. create a "Audience" protocol
   mapper on the client, included in the access token.
4. **User**: create a user (set a non-temporary password) and give it
   a realm role, e.g. `admin`, if you want gateway admin rights.

## 2. Point the gateway at Keycloak

Set these on the gateway (edit `deploy/openshift/secret.example.yaml`
→ copy to real values, and the `goalway-config` ConfigMap):

```yaml
# goalway-config (ConfigMap)
AUTH_PROVIDERS: "oidc,api_key"     # bearer tokens AND API keys both work

# goalway-secret (Secret)
OIDC_ISSUER: "https://<keycloak-route>/realms/gateway"
OIDC_AUDIENCE: "goalway"
OIDC_ADMIN_ROLES: "admin,administrator,gateway-admin"
```

Then restart/reload:

```bash
oc rollout restart deployment/goalway
```

`OIDC_JWKS_URL` is optional — it defaults to
`<OIDC_ISSUER>/.well-known/jwks.json`.

## 3. Get a token (password grant)

```bash
KC_URL="https://$(oc get route keycloak -o jsonpath='{.spec.host}')"

TOKEN=$(curl -s -X POST "$KC_URL/realms/gateway/protocol/openid-connect/token" \
  -d grant_type=password \
  -d client_id=goalway \
  -d username=demo \
  -d password=demo \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

# call the gateway with it
curl -s http://$(oc get route goalway -o jsonpath='{.spec.host}')/health \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

## Notes

- Tokens are validated locally against the realm JWKS; the gateway
  never calls Keycloak per request.
- If tokens lack the `goalway` audience, set `OIDC_AUDIENCE` to the
  client ID anyway and make sure the audience mapper is in place —
  otherwise validation fails with a 401.
