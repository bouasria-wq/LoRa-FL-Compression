# ============================================================
# CONFIG — ONLY FILE YOU NEED TO EDIT IN LAB
# Change these 3-4 lines per PC, nothing else
# ============================================================
ROLE      = "home"           # change to "server" on PC 5
HOME_ID   = 1                # change to 2, 3, 4 on each PC
USRP_TYPE = "b200"           # b200 PCs 1,2,3 | b210 PC 4 | b210 PC 5 (X410 acts as b210)
USRP_IP   = None             # None for all devices including server
SERVER_IP = "192.168.1.100"  # change to real server PC IP in lab
home_ips = {
    1: "192.168.1.101",      # change to real PC 1 IP in lab
    2: "192.168.1.102",      # change to real PC 2 IP in lab
    3: "192.168.1.103",      # change to real PC 3 IP in lab
    4: "192.168.1.104",      # change to real PC 4 IP in lab
}
