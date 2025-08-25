import pandas as pd

# Đọc file CSV
df = pd.read_csv("log_time.csv")

last_col = df.iloc[:, -1]

# Tính tổng theo giây
total_ms = last_col.sum()
total_sec = total_ms / 1000

print(f"Tổng thời gian: {total_ms} ms")
print(f"Tổng thời gian: {total_sec:.2f} s")