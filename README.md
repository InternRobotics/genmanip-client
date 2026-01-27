# genmanip-client

Client utilities for connecting to the GenManip eval server.

## Install (editable)

```bash
cd standalone_tools/packages/genmanip_client
pip install -e .
```

Optional deps for decoding tensors/images/numpy arrays:

```bash
pip install -e ".[full_numpy1]"
# OR
pip install -e ".[full_numpy2]"
```

## Usage

```bash
genmanip-client --host 127.0.0.1 --port 8087 --worker_ids 0,1
```

