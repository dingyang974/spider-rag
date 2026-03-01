import chardet

with open('./data/comments.csv', 'rb') as f:
    raw = f.read(10000)
    result = chardet.detect(raw)
    print(f"Detected encoding: {result}")

with open('./data/comments.csv', 'rb') as f:
    print("First 500 bytes:", f.read(500))
