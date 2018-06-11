import os, psutil

def usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / float(2 ** 30)

def mem_usage():
    print("Mem: ",usage(), "GB")
