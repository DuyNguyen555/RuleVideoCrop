import pandas as pd

# Đọc file CSV
df = pd.read_csv("output\part_0120\log_time.csv")

df_ms = df.drop(columns=["frame"])

# Tổng (ms và giây)
sum_ms = df_ms.sum()
sum_sec = sum_ms / 1000

# Trung bình (ms và giây)
mean_ms = df_ms.mean()
mean_sec = mean_ms / 1000

# Gộp thành 1 DataFrame thống kê
stats = pd.DataFrame({
    "sum_ms": sum_ms,
    "sum_sec": sum_sec,
    "mean_ms": mean_ms,
    "mean_sec": mean_sec
})

print(stats)