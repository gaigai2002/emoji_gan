import os
import pandas as pd
import base64


def prepare_emoji_images(csv_path="data/emoji.csv", output_dir="data/emoji"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 选择提取Apple的表情
    brand = "Apple"  # 也可以改成 Google、Facebook 等

    for i, row in df.iterrows():
        b64_str = row[brand]
        if isinstance(b64_str, str) and b64_str.startswith("data:image"):
            try:
                img_data = b64_str.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                img_path = os.path.join(output_dir, f"{i:04d}_{brand}.png")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
            except Exception as e:
                print(f"跳过第 {i} 行: {e}")

    print(f"✅ 提取完成，共生成 {len(os.listdir(output_dir))} 张图片")


if __name__ == "__main__":
    prepare_emoji_images("data/emoji.csv", "data/emoji")
