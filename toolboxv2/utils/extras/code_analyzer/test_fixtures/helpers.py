def compute(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

def transform(data):
    return [x * 2 for x in data]

def aggregate(data):
    return sum(data)

def unused_func():
    pass
