class Provider:
    def build_payload(self, req):
        raise NotImplementedError
    def parse_response(self, data):
        raise NotImplementedError