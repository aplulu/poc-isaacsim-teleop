# pi0-FASTファインチューニング用データ収集

Isaac Sim上でVRテレオペした操作を**LeRobot形式（codebase v2.1）** で保存し、openpiのpi0-FASTをファインチューニングするまでの手順です。

## 収集されるデータ

保存先は`datasets/<--dataset>/`です（既定は`datasets/crx10ial_teleop`）。

| キー | 形状 | 内容 |
| --- | --- | --- |
| `observation.images.base` | 256×256×3 (AV1動画) | 第三者視点（`/World/Cameras/Dataset`、カメラ揺らぎDRの対象） |
| `observation.images.wrist` | 256×256×3 (AV1動画) | フランジ上のリストカメラ |
| `observation.state` | float32[7] | J1..J6 [rad]、グリッパ開度（0 = 開 … 1 = 閉） |
| `action` | float32[7] | J1..J6の指令値[rad]（IKの出力）、グリッパ指令（0 / 1） |
| `observation.ee_pose` | float32[7] | ロボットベース座標系のTCP姿勢`[x y z qw qx qy qz]` |
| `action.ee_pose` | float32[7] | 指令TCP姿勢（同上） |
| `task` | 文字列 | 英語の指示文（例: `put the socket in the tray`） |

- サンプリングはシミュレーション時間基準の15 Hz（`--record-fps`）です。`action[t]`は`observation[t]`を取得した直後に出した指令なので、「観測tを見て方策が出すべき行動」とそのまま対応します。
- `meta/episodes_extra.jsonl`はLeRobot本体の形式外の付加情報です（LeRobotのローダは無視します）。成否、終了理由、物体の初期位置、DR設定、テレオペ設定を記録しています。
- LeRobot 0.3.3で読み込めることを確認済みです（`server/tools/check_lerobot.py`）。LeRobot 0.4以降（v3.0形式）で使う場合は、LeRobot付属のv2.1 → v3.0変換スクリプトを実行してください。

## 収集の流れ

1. Web Appの「データ収集」でタスク（例: 「◯◯をトレイに入れる」）と対象物体を選びます。「ランダム」を選ぶと毎回異なる物体が選ばれます。
2. 「● 録画開始」（VRでは反対の手のスティック押込、またはVRパネルのボタン）を押します。
   - リセットとDR（「開始時にDR」がONの場合）が行われ、0.8秒待機して「操作待ち」になります。
   - グリップ（クラッチ）を握るかグリッパを動かした瞬間から録画が始まります（「操作開始で録画」）。
3. タスクを実行します。
   - 自動成功判定がONの場合、成功条件（物体がトレイ内にあり、グリッパが開いている状態が0.6秒続く）を満たすと自動で保存されます。
   - 手動の場合は「成功で保存」「失敗で終了」「破棄」を押します。
   - 既定では失敗エピソードは保存しません（「失敗も保存」をONにすると`success=false`付きで保存します）。
4. 「保存後に次を自動開始」をONにすると、保存後に次のエピソードが自動で準備されるので連続して収集できます。
5. 誤って保存したエピソードは「最後のエピソードを削除」で取り消せます。

タスクテンプレートは`server/teleop/tasks.py`で追加できます（指示文、対象物体、成功条件）。

## データの確認

```bat
C:\isaacsim\python.bat server\tools\validate_dataset.py datasets\crx10ial_teleop
```

LeRobot本体での読み込み確認（WSL / Linux）:

```bash
uv venv .venv && . .venv/bin/activate
uv pip install --extra-index-url https://download.pytorch.org/whl/cpu "lerobot==0.3.3" torch torchvision "av<14"
python -P server/tools/check_lerobot.py datasets/crx10ial_teleop
```

## openpi (pi0-FAST)での学習

1. データセットをLeRobotのキャッシュに置きます（`repo_id = local/crx10ial_teleop`とする例）。

   ```bash
   mkdir -p ~/.cache/huggingface/lerobot/local
   cp -r /mnt/d/Projects/isaacsimvr/datasets/crx10ial_teleop ~/.cache/huggingface/lerobot/local/
   ```

2. openpiの`src/openpi/training/config.py`にデータ設定と学習設定を追加します。入出力の変換はLIBERO用のもの（base / wristの2画像、7次元行動）をそのまま流用できます。関節角は絶対値で保存しているので、DROIDやALOHAの設定と同様に、腕の6軸だけを差分行動に変換します。

   ```python
   @dataclasses.dataclass(frozen=True)
   class LeRobotCrxDataConfig(DataConfigFactory):
       @override
       def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
           repack = _transforms.Group(inputs=[_transforms.RepackTransform({
               "observation/image": "observation.images.base",
               "observation/wrist_image": "observation.images.wrist",
               "observation/state": "observation.state",
               "actions": "action",
               "prompt": "prompt",
           })])
           data_transforms = _transforms.Group(
               inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
               outputs=[libero_policy.LiberoOutputs()],
           )
           mask = _transforms.make_bool_mask(6, -1)  # J1..J6 -> delta, gripper stays absolute
           data_transforms = data_transforms.push(
               inputs=[_transforms.DeltaActions(mask)],
               outputs=[_transforms.AbsoluteActions(mask)],
           )
           return dataclasses.replace(
               self.create_base_config(assets_dirs, model_config),
               repack_transforms=repack,
               data_transforms=data_transforms,
               model_transforms=ModelTransformFactory()(model_config),
               action_sequence_keys=("action",),
           )

   TrainConfig(
       name="pi0_fast_crx10ial",
       model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
       data=LeRobotCrxDataConfig(
           repo_id="local/crx10ial_teleop",
           base_config=DataConfig(prompt_from_task=True),
       ),
       weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
       num_train_steps=30_000,
   ),
   ```

   `LiberoInputs`の引数はopenpiのバージョンによって異なるので、手元の`libero_policy.py`に合わせてください。

3. 正規化統計を計算してから学習します。

   ```bash
   uv run scripts/compute_norm_stats.py --config-name pi0_fast_crx10ial
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_crx10ial --exp-name=crx_v1
   ```

## 収集のコツ

- 1タスクあたり50〜200エピソードが目安です。物体の初期配置DRは毎回、照明・カメラ揺らぎ・鏡の有無は「開始時にDR」で有効にした項目だけが適用されます。
- 動作はゆっくり、迷いなく行ってください。グリップを離して持ち替える時間（クラッチOFF）は行動が止まった区間として記録されます。
- 2F-85の最大開口は85 mmです。赤いリンゴ（約78〜81 mm）は余裕が少ないので、真上から中心を狙ってください。
