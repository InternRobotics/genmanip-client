# genmanip-replay-v1

The protocol adds two endpoints to the existing authenticated evaluation API.
It uses the same trusted binary/pickle channel as the existing eval endpoints;
it does not change authentication or make pickle safe for untrusted input.

## Discovery

`GET /request_protocol` returns JSON:

```json
{"protocol":"genmanip-replay-v1","epoch":"server-instance-id","worker_generations":{"0":1}}
```

An absent worker generation is zero. The server epoch changes on process restart
and job replacement. Each worker generation changes before a kill or worker
replacement. Discovery must not be cached by a proxy. A client discovers once per
stream, never refreshes the epoch/generation to retry an ambiguous action, and
creates a new stream only after explicitly resetting the environment.

## Submission and retrieval

`POST /reliable_request` has `Content-Type: application/octet-stream` and a pickled
dictionary containing:

- `epoch`: discovered server epoch.
- `worker_generations`: discovered generation for each worker in this stream.
- `stream_id`: random UUID hex string, 32 characters.
- `sequence`: positive integer starting at 1, incremented only for a new operation.
- `worker_ids`: fixed list of string IDs owned by this stream.
- `operation`: `step`, `step_chunk`, `reset`, or `reset_result`.
- `payload`: bytes of the original pickled operation body.

Bearer authentication and `run_id` routing are sent as for legacy requests.
The same request ID is `(epoch, stream_id, sequence)`; all retransmissions use
identical envelope bytes and rendering options. Each stream executes sequentially.
The server must register the request before dispatching simulation work and keep
a strong reference to the task. HTTP disconnects and polling timeouts must not
cancel it.

After at most five seconds waiting for a result, the server returns HTTP 202 with
a pickled envelope containing `protocol`, `epoch`, `stream_id`, `sequence`, and
`state="pending"`. The client resends the same request to retrieve its result.
Completion is HTTP 200 with the same identifiers, `state="completed"`,
`status_code=200`, and `body` containing the original binary response bytes.

Simulation failures are cached completion envelopes with their original
`status_code` and `detail`, sent as HTTP 200 so a proxy cannot erase their detail.
They are terminal; the operation must not be run again. Protocol violations use
HTTP 200 with `protocol`, `state="rejected"`, and `detail` and are also terminal.
HTTP 408, 429, 450, 500, 502, 503, 504 or a lost/truncated response may be retried
with the same ID, subject to the network recovery budget.

## Required safety guarantees

- Repeated in-flight IDs share one task; repeated completed IDs return one cached
  result, including failures. Reusing an ID with different content is rejected.
- Keep the latest response per stream and a permanent sequence high-water mark
  for that server epoch. Reject older sequences, gaps, and advancing while pending.
- Retain retired stream IDs as tombstones. At capacity, reject new streams rather
  than silently evicting IDs and allowing actions to execute twice.
- Reject a mismatched server epoch or worker generation, including a delayed
  first request that arrives after worker replacement.
- Workers are owned by a stream. Reject overlapping streams and legacy simulation
  calls on those workers; lifecycle operations invalidate affected streams.
- A simulator timeout does not prove that the executor stopped. Cache the failure
  and refuse further actions on that stream until an explicit environment reset.

The paired server implementation keeps this state in one ASGI process. Multiple
replicas need shared admission/result storage or routing that preserves the owner;
this in-memory protocol does not provide durable recovery across process crashes.

Client logging uses `genmanip_client.reliable_transport` and includes the stable
`stream_id:sequence` ID. Enable INFO logging for completion/attempt counts; retry
warnings are emitted by default. Server traces can include the same `replay_id`.
