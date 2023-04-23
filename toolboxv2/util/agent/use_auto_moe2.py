
GLOBAL = 0
def kp(x):
    global GLOBAL
    GLOBAL += 1
    return "KP "+str(GLOBAL)+ " " +x

def teste(response):
    return (
        kp(
            line
        )
        for line in response
    )


print("hallo".split(","))


