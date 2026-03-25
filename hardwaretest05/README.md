# hardwaretest05 - ME-CFL with 3 Homes + Server over USRP B200 (TDMA)

## What's Different from hardwaretest04

| Version | Homes | USRPs | Scheduling |
|---------|-------|-------|-----------|
| hardwaretest04 | 1 home + 1 server | 2 USRPs | None needed |
| **hardwaretest05** | **3 homes + 1 server** | **4 USRPs** | **TDMA time-slotted** |

## Hardware Required

- **4x Ettus USRP B200** connected via USB 3.0
- **4x LoRa antennas** (868 MHz) attached to TX/RX ports
- Lab PC with 4 USB 3.0 ports (or USB hub)

### USRP Assignment
| USRP | Serial | Role |
|------|--------|------|
| B200 #1 | (run uhd_find_devices) | Home 1 TX |
| B200 #2 | (run uhd_find_devices) | Home 2 TX |
| B200 #3 | (run uhd_find_devices) | Home 3 TX |
| B200 #4 | (run uhd_find_devices) | Server RX + TX |

Get serials: `uhd_find_devices`

## TDMA Slot Layout (per day)

```
Time:   0s          30s         60s         90s         120s
        |-----------|-----------|-----------|-----------|------>
        | Home 1 TX | Home 2 TX | Home 3 TX | Server   | Homes
        |           |           |           | aggregate | receive
        |           |           |           | + broadcast| global
```

- Each home waits for its assigned slot before transmitting
- Server RX stays open for the full 90s window
- No RF collisions between homes

## How to Run

### Step 1: Find your USRP serials
```bash
uhd_find_devices
```

### Step 2: Start server (Terminal 1)
```bash
cd ~/LoRa_Fl_Compression.proj/hardwaretest05
python3 server_aggregator.py --n_homes 3 --days 7 \
    --rx_serial <SERVER_RX_SERIAL> \
    --tx_serial <SERVER_TX_SERIAL>
```

### Step 3: Start all 3 homes (Terminals 2-4)
```bash
# Terminal 2
python3 home_node.py --home_id 1 --days 7 --epochs 100 \
    --tx_serial <HOME1_TX_SERIAL>

# Terminal 3
python3 home_node.py --home_id 2 --days 7 --epochs 100 \
    --tx_serial <HOME2_TX_SERIAL>

# Terminal 4
python3 home_node.py --home_id 3 --days 7 --epochs 100 \
    --tx_serial <HOME3_TX_SERIAL>
```

### Or start all 3 homes at once:
```bash
python3 home_node.py --home_id 1 --tx_serial SERIAL1 --days 7 --epochs 100 &
python3 home_node.py --home_id 2 --tx_serial SERIAL2 --days 7 --epochs 100 &
python3 home_node.py --home_id 3 --tx_serial SERIAL3 --days 7 --epochs 100 &
wait
```

## Key Metrics Reported

- **MAE** — prediction error per home per day
- **Accuracy** — model accuracy per home
- **PDR** — Packet Delivery Ratio (RF packets successfully received / total)
- **ToA** — Time on Air per LoRa packet
- **Zeta** — heterogeneous variance across homes

## Files

```
hardwaretest05/
├── home_node.py              # TDMA-aware home node
├── server_aggregator.py      # TDMA server (listen all slots + aggregate)
├── compression/
│   ├── hegazy.py             # Hegazy compression with error feedback
│   └── hegazy_lora_bridge.py # struct.pack + ASCII encoding
├── local_home/
│   ├── load_data.py, model.py, train.py
├── server/
│   └── aggregate.py          # ME-CFL aggregation
├── lora/
│   └── gr_lora_usrp.py       # TDMA-aware USRP bridge (envelope only)
└── data/
    └── home_01.csv ... home_10.csv
```
