# ============================================================
# CONFIG — ONLY FILE YOU NEED TO EDIT IN LAB
# Change these 3-4 lines per laptop, nothing else
# ============================================================

ROLE      = "home"           # change to "server" on laptop 5
HOME_ID   = 1                # change to 2, 3, 4 on each laptop
USRP_TYPE = "b200"           # b200 laptops 1,2,3 | b210 laptop 4 | x410 laptop 5
USRP_IP   = None             # only needed for x410 server — set to "192.168.10.2"
SERVER_IP = "192.168.1.100"  # change to real server laptop IP in lab

home_ips = {
    1: "192.168.1.101",  # change to real laptop 1 IP in lab
    2: "192.168.1.102",  # change to real laptop 2 IP in lab
    3: "192.168.1.103",  # change to real laptop 3 IP in lab
    4: "192.168.1.104",  # change to real laptop 4 IP in lab
}
