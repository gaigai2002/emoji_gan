from src.train import train

if __name__ == "__main__":
    train(data_dir="data/emoji",
          epochs=200,  # ✅ 增加训练轮数
          batch_size=128,  # ✅ 稍微大一点更稳定
          z_dim=100,
          lr=0.0001)  # ✅ 稍微降低学习率，稳定训练

