import time
import random
def traceroute_stimulation(host):
    print(f"tracing route to {host}...\n")
    max_hops=10
    for hop in range(1,max_hops+1):
        if random.random()<0.15:
            print(f"{hop}\t*\t Request Timeout...")
        else:
            delay=round(random.uniform(10,100),2)
            router_ip=f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
            print(f"{hop}\t{router_ip}\t{delay}ms")
            time.sleep(1)
    print("Traceroute completed")
host="google.com"
traceroute_stimulation(host)
