# hardwaretest11 - ME-CFL with 3 Homes + Server over USRP B200 (TDMA)

## What's Different from hardwaretest10

| Version | Homes | USRPs | Scheduling |
|---------|-------|-------|-----------|
| hardwaretest10 | 1 home + 1 server | 2 USRPs | Flag handshake |
| **hardwaretest11** | **3 homes + 1 server** | **4 USRPs** | **TDMA time-slotted** |

## USRP Assignment

Edit gr_lora_usrp.py and replace these placeholders with actual serials from uhd_find_devices:

| Placeholder | Role |
|-------------|------|
| SERIAL_HOME1_TX | Home 1 TX USRP |
| SERIAL_HOME2_TX | Home 2 TX USRP |
| SERIAL_HOME3_TX | Home 3 TX USRP |
| SERIAL_SERVER | Server USRP (TX broadcast) |

Run: uhd_find_devices

## TDMA Slot Layout per Day

    Server waits for all 3 homes ready
    -> Signal Home 1 slot -> Home 1 TX -> Server listening
    -> Signal Home 2 slot -> Home 2 TX -> Server listening
    -> Signal Home 3 slot -> Home 3 TX -> Server listening
    -> Server aggregates
    -> Server TX broadcasts global model
    -> All homes read global_model_day{N}.bin
    -> Next day

## How to Run

Terminal 1 - Server:
    cd hardwaretest11
    python3 server_aggregator.py --n_homes 3 --days 7

Terminal 2 - Home 1:
    python3 home_node.py --home_id 1 --days 7 --epochs 100 --tx_serial SERIAL_HOME1_TX

Terminal 3 - Home 2:
    python3 home_node.py --home_id 2 --days 7 --epochs 100 --tx_serial SERIAL_HOME2_TX

Terminal 4 - Home 3:
    python3 home_node.py --home_id 3 --days 7 --epochs 100 --tx_serial SERIAL_HOME3_TX

## Key Features

- CRC validation built into LoRa PHY (Chen's code)
- TDMA prevents RF collisions between homes
- Flag-based handshake prevents USRP conflicts
- File backup ensures global model delivery even if RF fails
- PDR reported per day in final summary
