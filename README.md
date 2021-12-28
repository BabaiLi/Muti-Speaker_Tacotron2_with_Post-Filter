# Muti-Speaker_Tacotron2_with_Post-Filter

# 從WAV檔案抓取梅爾頻譜
```
python3 extract_mel_util.py <input> --output_dir <output_path> # -> target_dirs
```
# 修改config.py參數
## 訓練英文以外的語言請用第一個，訓練英文則用第二個
```
text_cleaners = ["transliteration_cleaners"] or ["english_cleaners"]
```
# 要將輸入文字轉換成CMU格式請提供字典位置
```
cmudict_path  = None
```
# 訓練集和測試集文件位置
```
train = <str>
val   = <str>
```
# 讀取npy檔案或是tensor檔案
```
load_from_numpy = True or False
```
# 若使用外部語者Embedding
```
spk_emb = str<path> or None
```
# 是否使用語者Embedding
```
use_spk_emb       = True or False
use_spk_table     = True or False  # 建立內部語者Embedding，無法擴展到沒看過語者
add_spk_to_prenet = True or False  # 在PreNet層加入語者Embedding，或許能夠加強語者相似度
```
# 訓練模型，模型和tensorboard存在'outdir'資料夾中
```
python3 train.py
```
# 若要使用額外的Post-Filter 請執行下列
## 將 train.txt 與 val.txt 文件合併為 total.txt
```
python3 train_diff_prepare.py <checkpoint_path> # -> mel_dirs
cd Diff
python3 train.py <output> --mel_dirs <mel_dirs> --target_dirs <target_dirs>
                                     # line 33                  line 5
```
# Inference, 音檔存在inference資料夾內
## 回到Tacotron 2資料夾
```
python3 inference.py -c <checkpoint_path> -d <Diff/output> -s <speaker> -e <audio_path>
```
