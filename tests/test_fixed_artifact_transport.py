import io
import json
import unittest

from generation.fixed_artifact_transport import ENDPOINT, openrouter_transport


class FixedTransportTests(unittest.TestCase):
    def test_no_credentials_or_network_on_construction_and_exact_request(self):
        seen=[]; credentials=[]
        raw={"id":"id", "model":"model", "provider":"provider", "usage":{"completion_tokens":2},
             "choices":[{"finish_reason":"length","message":{"content":"partial"}}]}
        def key(): credentials.append(1); return "fake-test-key"
        def open_(req,timeout):
            seen.append((req,timeout)); return io.BytesIO(json.dumps(raw).encode())
        send=openrouter_transport(key,opener=open_)
        self.assertEqual(seen,[]);self.assertEqual(credentials,[])
        result=send({"model":"model","messages":[{"role":"user","content":"α"}]})
        self.assertEqual(len(seen),1);self.assertEqual(credentials,[1])
        self.assertEqual(seen[0][0].full_url,ENDPOINT)
        self.assertEqual(json.loads(seen[0][0].data)["messages"][0]["content"],"α")
        self.assertEqual(result["finish_reason"],"length")
        self.assertEqual(result["content"],"partial")
        self.assertNotIn("fake-test-key",json.dumps(result))

    def test_transport_error_has_no_retry(self):
        calls=[]
        def fail(*args,**kwargs): calls.append(1);raise OSError("offline fixture")
        send=openrouter_transport(lambda:"fake",opener=fail)
        with self.assertRaises(OSError):send({})
        self.assertEqual(calls,[1])

    def test_empty_key_does_not_send(self):
        calls=[]
        send=openrouter_transport(lambda:"",opener=lambda *a,**k:calls.append(1))
        with self.assertRaises(ValueError):send({})
        self.assertEqual(calls,[])

    def test_invalid_response_rejected(self):
        for response in ({"choices":[]},{"choices":[{"message":None}]},[] ):
            send=openrouter_transport(lambda:"fake",opener=lambda *a,**k:io.BytesIO(json.dumps(response).encode()))
            with self.assertRaises(ValueError):send({})


if __name__ == '__main__': unittest.main()
