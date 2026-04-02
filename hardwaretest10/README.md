# hardwaretest10 - ME-CFL with Real USRP B200 using Chen's LoRa Code

## What's Different

| Version | TX Method |
|---------|-----------|
| hardwaretest04 | EPFL lora_TX.grc compiled with grcc, message strobe patched |
| **hardwaretest10** | **Chen's lora_TX.py directly, payload posted to whitening block message port** |

## Key Approach

Uses Chen's `run_lora_transmitter.py` approach:
- Starts lora_TX flowgraph
- Disables default message strobe (set_frame_period to very large value)
- Posts our compressed payload directly to whitening block message port via pmt
- No file source, no patching, no grcc needed

## Hardware Required

- 2x Ettus USRP B200
- TX USRP serial: 32BBAD0
- RX USRP serial: 32BBAD9

## How to Run

Terminal 1 - Server:
    cd hardwaretest10
    python3 server_aggregator.py --n_homes 1 --days 7

Terminal 2 - Home:
    cd hardwaretest10
    python3 home_node.py --home_id 1 --days 7 --epochs 100

## Files

- lora/lora_TX.py — Chen's TX flowgraph (untouched except serial)
- lora/lora_RX.py — Chen's RX flowgraph (untouched except serial)
- lora/gr_lora_usrp.py — bridge that posts payload to whitening block
- compression/ — Hegazy compression
- local_home/ — LSTM model, training, data loading
- server/ — ME-CFL aggregation
