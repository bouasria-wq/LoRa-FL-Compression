# hardwaretest03 - ME-CFL with Real GNU Radio Companion Flowgraphs

## What's Different

| Version | LoRa Transport |
|---------|---------------|
| hardwaretest02 | Python builds USRP flowgraphs programmatically |
| onlinetest02 | Python builds simulated TX chain programmatically |
| **hardwaretest03** | **Real .grc flowgraph from GNU Radio Companion GUI** |

## How It Works

1. `tx_rx_sim.grc` is the **real GNU Radio Companion flowgraph file**
2. You can open it in the GUI: `gnuradio-companion lora/tx_rx_sim.grc`
3. Python uses `grcc` to compile `.grc` → `.py` (same as clicking Generate/F5 in GUI)
4. The compiled flowgraph runs the full IEEE-standard LoRa PHY:
   - TX: Whitening → Header → CRC → Hamming → Interleaver → Gray Demap → CSS Modulate
   - Channel: AWGN noise model
   - RX: Frame Sync (CFO/STO est.) → FFT Demod → Gray Map → Deinterleave → Hamming Dec → Header Dec → Dewhiten → CRC Verify

## To View in GUI

```bash
conda activate lora
gnuradio-companion hardwaretest03/lora/tx_rx_sim.grc
```

## To Run

Terminal 1 — Server:
```bash
cd hardwaretest03
python server_aggregator.py --n_homes 1 --days 7
```

Terminal 2 — Home node:
```bash
cd hardwaretest03
python home_node.py --home_id 1 --days 7 --epochs 100
```

## Switching to USRP B200 Hardware

Open `tx_rx_sim.grc` in GNU Radio Companion, then:
1. Delete the `Channel Model` block
2. Add `UHD: USRP Sink` after Modulate (TX output)
3. Add `UHD: USRP Source` before Frame Sync (RX input)
4. Save as `tx_rx_usrp.grc`

## Requirements

```bash
conda activate lora
conda install -c tapparelj -c conda-forge gnuradio-lora_sdr
pip install tensorflow pandas numpy scipy
```
