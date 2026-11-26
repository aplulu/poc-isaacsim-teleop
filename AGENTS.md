# AGENTS.md

コーディングエージェント向けの作業メモです。使い方・操作方法は[README.md](README.md)、データ収集と学習は[docs/vla_finetuning.md](docs/vla_finetuning.md)を参照してください。

## 概要

Quest 3などの内蔵ブラウザ（WebXR）から、Isaac Sim 6.0.1上のFANUC CRX-10iA/L + Robotiq 2F-85を遠隔操作します。映像はWebRTC、操作入力はWebRTCのDataChannelです。LeRobot v2.1形式での録画、操作者アバター（VRM）、ドメインランダマイゼーションを含みます。

- `server/`: Isaac SimのPython（`SimulationApp`）上で動くaiohttp + aiortcサーバ
- `webapp/`: Vite + React + React Three Fiber（@react-three/xr）+ Zustand。`webapp/dist`をサーバが配信

## 環境と作業上の注意

- Windows 11、Isaac Sim 6.0.1は`C:\isaacsim`（Python 3.12）、GPUはRTX 3090。**gitリポジトリではありません**（変更前のバックアップは自分で取る）。
- Pythonは必ず`C:\isaacsim\python.bat`（Isaac Simの環境）を使います。Isaac Simを起動しない単体スクリプトは`C:\isaacsim\kit\python\python.exe`でも動きます。PATH上の`python`はWindowsのスタブで、何も実行しません。
- 追加のPythonパッケージは`server\_deps`に入れます（`server\install_deps.bat`、`server/requirements.txt`）。`C:\isaacsim`には入れません。単体スクリプトでは`sys.path.append(server/_deps)`と、PyAVのために`os.add_dll_directory(server/_deps/av.libs)`が要ります。
- ファイルのダウンロードやパッケージの追加は、ユーザーの許可を得てから行います。
- ポート8443では、ユーザーが自分でサーバ（GUI版）を動かしていることがあります。テスト用のサーバを起動する前に確認し、ユーザーのサーバは止めないでください。テストのペアリング要求がユーザーのサーバに届くと、そのウィンドウに要求が表示されます。
- `datasets/`はユーザーが集めたデータです。テストの録画で汚さないでください（`--no-record`や別の`--dataset`を使う）。
- サーバのログは`server/logs/teleop.log`。ペアリングコードもここに出るので、テストツールはここから読みます。コンソール出力をリダイレクトするときは`python.bat -u`にしないとバッファされて止まって見えます。
- 内蔵ブラウザは自己署名証明書を受け付けません。UIを確認するときは`--no-tls --port 8080`で起動します。

## ビルドと実行

```bat
run_server.bat                                         :: 依存・シーン・Web App を用意してから起動
C:\isaacsim\python.bat server\build_scene.py           :: シーン USD の再生成（グリッパ・鏡・小物を変えたとき）
C:\isaacsim\python.bat server\main.py --headless       :: テスト用（ペアリングコードはコンソールとログ）
cd webapp && npm run build                             :: tsc の型検査 + vite build（dist をサーバが配信）
cd webapp && npx tsc --noEmit -p .                     :: 型検査のみ
```

`main.py`の主なオプションはREADMEの「セットアップと起動」にあります。開発時の追加オプション: `--viewport auto|on|off`、`--kit-trace <出力.json>`、`--profile`、`--video-codec auto|nvenc|x264|vp8`、`--no-tls`。

## 構成

```
 Quest Browser (WebXR)                              Windows PC (Isaac Sim 6.0.1)
┌──────────────────────────┐   HTTPS (REST/JWT)   ┌─────────────────────────────────────┐
│ Vite + React + R3F       │ ───────────────────▶ │ aiohttp (別スレッド)                  │
│ @react-three/xr, Zustand │   WebRTC             │  ├ /api/pair/*   6桁コード → JWT 発行  │
│  ├ ペアリング画面         │ ◀── video (H264) ─── │  ├ /api/webrtc/offer (aiortc)          │
│  ├ VR モニター/操作パネル │ ── DC "teleop" ────▶ │  └ /api/command  リセット/DR/カメラ     │
│  └ コントロールパネル     │ ◀─ DC "events" ────▶ │ SimulationApp (メインスレッド)          │
└──────────────────────────┘                      │  ├ scene/vr_teleop_scene.usda          │
                                                  │  ├ DLS IK + クラッチ式テレオペ          │
                                                  │  ├ ドメインランダマイゼーション          │
                                                  │  └ omni.ui ペアリングウィンドウ          │
                                                  └─────────────────────────────────────┘
```

- **スレッド**: aiohttp / aiortcは別スレッドのイベントループ。シミュレーション（Kit更新、IK、アバター、画像取得）はメインスレッド。両者は`bridge.py`でつなぎます（`TeleopInput`は最新のXRサンプル1件をロック付きで保持、`CommandQueue`はコマンドをメインスレッドで実行、`FrameHub`は映像フレーム / H.264パケットを各接続へ、`StateHub`は状態）。GPUからのエンコードは専用スレッド（`gpu-h264`）。
- **メインループ**（`main.py`）: 前処理（入力・IK・アバター）→ Kit更新（物理 + 描画）→ 後処理（画像取得・配信・録画）。

### ファイル

```
run_server.bat
scene/vr_teleop_scene.usda       生成されるシーン
server/
  build_scene.py                 シーン生成
  main.py                        Isaac Sim 起動 + サーバ、メインループ
  install_deps.bat / requirements.txt   server/_deps への依存インストール
  teleop/
    config.py                    アセット/レイアウト/サーバ設定（ServerConfig、機能フラグ）
    auth.py                      6桁コード + JWT
    http_server.py               aiohttp (REST, 静的配信, TLS)、ALLOWED_COMMANDS
    webrtc.py                    aiortc (映像トラック, DataChannel, 操作権)
    vmc.py                       VMC Protocol（OSC）のモーション受信 → XR サンプル
    sim.py                       IK・テレオペ・グリッパ・カメラ・状態・解像度・配信
    randomization.py             ドメインランダマイゼーション（照明/素材/カメラ/鏡/物理/配置）
    recorder.py                  エピソード録画（状態機械・成功判定・学習用カメラ）
    lerobot_writer.py            LeRobot v2.1 ライタ（parquet + AV1、統計量）
    tasks.py                     言語タスクテンプレートと成功条件
    immersive.py                 没入表示（頭部追従ステレオカメラ、アンカー、移動）
    vrm.py                       VRM → USD 変換（UsdSkel、材質、スプリングボーン情報）
    avatar.py                    操作者アバターの全身 IK・スプリングボーン・当たり判定の姿勢
    avatar_ik.py / avatar_springs.py  アバターの IK / FK・揺れもの・回転出力の Warp カーネル
    nvenc.py                     NVENC の H.264 エンコーダ（PyAV 経由）と、H.264 を優先するコーデック選択
    gpu_stream.py                GPU の画像をそのまま NVENC へ（PyNvVideoCodec、全接続で共有）
    perf.py                      計測値の集計（PERF）
    ui_window.py                 omni.ui ペアリングウィンドウ
    certs.py / bridge.py / mathutil.py
server/tools/                    確認・計測ツール（下記）
docs/vla_finetuning.md           データ収集と openpi (pi0-FAST) 学習手順
assets/props/README.md           自前の小物の追加方法
webapp/src/
  store.ts (Zustand, 永続化する設定は partialize) / api.ts / webrtc.ts / types.ts / vmc.ts / perfStats.ts
  components/PairingScreen.tsx, ControlPanel.tsx, RecordingPanel.tsx
  xr/XRScene.tsx, VideoScreen.tsx, ImmersiveView.tsx, TeleopSender.tsx, VRPanel.tsx, frameMeta.ts
```

## テストと確認ツール

ヘッドレスのサーバ（`C:\isaacsim\python.bat server\main.py --headless`）を起動してから実行します。いずれも`server/logs/teleop.log`からペアリングコードを読み、自分でペアリングします。

| ツール | 内容 |
| --- | --- |
| `test_client.py --log server\logs\teleop.log` | ペアリング → 不正トークン拒否 → JWT → WebRTC映像受信 → クラッチ移動 → DR → カメラ切替 → リセット → ログアウト後の失効。フレームを`server/tools/_out`に保存 |
| `api_cmd.py` | 任意のコマンドを実行 |
| `scripted_episode.py` | API経由でピック&プレースを自動実行し、データセット生成を確認 |
| `validate_dataset.py` / `check_lerobot.py` | 保存したデータセットの検証 |
| `mapping_check.py` / `stereo_check.py` / `immersive_check.py` | 操作基準・立体視・没入表示 |
| `avatar_check.py` / `avatar_collision_check.py` / `avatar_springs_check.py` | アバター（姿勢、当たり判定、Warpとnumpy参照実装の一致と時間） |
| `hand_grab_check.py` | 没入表示で手で掴む、解像度変更 |
| `perf_check.py` | 表示モードごとの時間の内訳、物理・配信カメラを止めた場合、没入表示の画像と姿勢のずれ。`--only mono,stereo,immersive,immersive-noavatar,lag`、`--immersive-size`、`--dlss`、`--kit-trace` |
| `physics_bench.py` / `kit_bench.py` | シーン単体の物理コスト |
| `kit_trace_report.py <trace.json>` | Kitプロファイラのトレースをメインスレッドのms/フレームの木で表示 |

- `perf_check.py`のlagテストは頭を90°振ったまま終わるので、次の接続のアンカーの向きがずれます。`avatar_collision_check.py` / `hand_grab_check.py`は、アンカーのyawが180°になるまで正面リセットを繰り返してから始めます。新しいツールでも同じ対策が要ります。
- Web App側は`npm run build`（型検査込み）が通ることを確認します。ヘッドセットなしのVR確認はURLに`?xremulate`（IWERのQuest 3エミュレータ）。

## 実装メモ

### シーンと物理

- `build_scene.py`がIsaac Sim / SimReadyアセットを参照して`scene/vr_teleop_scene.usda`を生成し、`main.py`は開くだけです。
- **小物の前処理**: 小物は単体の剛体として扱います。Isaac LabのFactoryボルト・ナットのように単体アーティキュレーションのものはアーティキュレーション設定を外し、ボルトをワールドに固定している関節（root_joint）も無効化します（そのままだと剛体ビューでの移動で物理が破綻します）。
- **掴む対象セット**: 選ばれていないセットはシーンの遠く（重力OFF）に退避します。自前のモデルは`assets/props/<セット>/`。
- **吹き飛び対策**: 腕の関節トルク上限J1..J6 = 150/150/100/30/30/20 N m。小物には最大めり込み解消速度0.3 m/s、CCD、高いソルバー反復回数。
- **把持アシスト**: 2F-85は荷重でmimic関節の追従が崩れて指パッドが傾き、小さい・丸い・平たい部品を押し出します。指が実際に動いてから物体に当たって止まった時点で、パッド間の物体をTCPに固定します（開けば離す）。指は接触位置で保持。
- **テーブル**: 天板は見た目と同じ寸法の見えない箱で判定（凸分解すると約2 cm厚くなり物体が浮くため）。TCPの下限は天板 + 3 cm（閉じるときに円弧を描いて下がる指先も当たらない）。
- **グリッパ**: 2F-85の`ArticulationRootAPI`を外し、`J6_link`に固定ジョイントで取り付けて腕と単一のアーティキュレーション（J1..J6 + `finger_joint`）。`build_scene.py --gripper virtual`で旧来の吸着式。
- **物理の高速化**: SimReady小物のSDFコリジョンを凸包に置き換え（トレイ・カップ・バナナは凸分解）。ロボットのソルバー反復32 → 12。
- **実時間**: `manualModeEnabled=false` / `useFixedTimeStepping=false`。物理は既定30 Hzでサブステップ（`--physics-hz`）。1フレームの物理ステップ数はフレーム時間 × 物理レートなので、60 Hzでは物理だけでループの約4割でした（30 Hzでfps約1.5倍）。

### 認証

- 6桁コードは暗号学的乱数で生成し、Isaac Simのウィンドウ（とログ）にのみ出します（HTTP応答には含めない）。有効期限120秒、誤入力5回で失効、IPごとのレート制限、要求元IPと確認元IPの一致を確認。同じIP・ブラウザからの再要求は前の要求を置き換えます。
- JWTはHS256（`iss/aud/sub/exp/jti`、既定8時間）。署名鍵は起動ごとに生成（再起動で全トークン無効）。RESTとWebRTCオファーは`Authorization: Bearer`で検証。Revokeで即時失効しWebRTCも切断。Web Appは401でペアリング画面へ。
- ログアウトでの接続の後始末はバックグラウンドで行います（同期で閉じると応答が数分止まったため）。

### API

| Method | Path | 認証 | |
| --- | --- | --- | --- |
| GET | `/api/info` | - | サーバ情報 |
| POST | `/api/pair/request` | - | `{deviceName}` → `{requestId}` |
| POST | `/api/pair/confirm` | - | `{requestId, code}` → `{token, expiresAt}` |
| GET | `/api/session` / POST `/api/session/logout` | JWT | |
| POST | `/api/webrtc/offer` | JWT | SDP offer → answer |
| GET | `/api/state` | JWT | 関節・TCP・グリッパ・物体位置・入力など |
| POST | `/api/command` | JWT | `{name, payload}`: `reset` `randomize` `get_dr_config` `set_dr_config` `set_camera` `set_gripper` `set_mirror` `go_home` `move_tcp` `set_teleop_settings` `pause` `resume` `set_resolution` `perf_probe`ほか / 録画: `task_templates` `rec_start` `rec_stop` `rec_discard` `rec_status` `rec_settings` `rec_delete_last`（全一覧は`http_server.py`の`ALLOWED_COMMANDS`） |
| GET | `/api/snapshot` | JWT | 現在のカメラ画像(JPEG) |

### DataChannel

| チャンネル | 設定 | 内容 |
| --- | --- | --- |
| `teleop` | `ordered: false, maxRetransmits: 0` | XRサンプル約60 Hz（`TeleopSender.tsx`）。`bufferedAmount`が16 KBを超えたら送らずに捨てる |
| `events` | 既定（順序保証・再送あり） | サーバ → 状態（20 Hz）・`fmeta`（フレームの姿勢）・`pong`・`cmd_result`。Web App → `ping`・`perf`・`take_control`・`cmd` |

- XRサンプル（JSON）: `{t:"xr", ts, head:{p,q}, left/right:{p,q,trigger,grip,axes}}`。`ts`はWeb Appの`performance.now()`。サーバは受信時刻`_rx`を付けて`TeleopInput.update()`に渡し、メインループは`latest(max_age=0.5)`で最新だけを使います。
- 操作権: 1台だけ。持ち主の接続が閉じたか1秒以上入力が途絶えたら、入力を送ってきた接続に移ります（ページ再読み込みで接続が一時的に複数できるため）。
- `cmd`はRESTの`/api/command`と同じ`ALLOWED_COMMANDS`だけを受け付けます。

### モーションのVMC Protocol形式

`teleop`チャンネルのXRサンプルは、既定でVMC Protocol（OSC）のバンドルです（`webapp/src/vmc.ts` → `server/teleop/vmc.py`）。1サンプル = 1バンドル（time tagはimmediate）。

- **形式の切り替え**: サーバの環境変数`TELEOP_MOTION_FORMAT`（`vmc`既定 / `json`、`config.MOTION_FORMAT`）を`/api/info`の`motionFormat`でWeb Appに伝え、Web Appはそれに従います（`vmc.ts`の`motionFormat()`。UIの切り替えやブラウザへの保存はなし）。`motionFormat`を返さないサーバ（VMC対応前）にはJSONで送ります。起動中の古いサーバも、ビルドし直した`webapp/dist`を配信するためです。
- **受信**: サーバは設定に関係なく常に両方を受け付けます（文字列 = JSON、バイナリ = OSC。形式が変わると`peer … motion format: …`をログ）。

| アドレス | 引数 | 内容 |
|---|---|---|
| `/VMC/Ext/OK` | `i` | 1（読み込み済み） |
| `/VMC/Ext/T` | `f` | 時刻[s]（`performance.now()/1000`） |
| `/VMC/Ext/Hmd/Pos` | `s fffffff` | `"HMD"`, 位置xyz, 回転xyzw |
| `/VMC/Ext/Con/Pos` | `s fffffff` | `"LeftController"` / `"RightController"`, グリップ姿勢 |
| `/VMC/Ext/Con` | `i s i i i fff` | active=2, 名前, isLeft, isTouch=0, isAxis=1, axis xyz。`Trigger`・`Grip`（axis.x）、`Stick`（axis.xy、上が+y） |

- 座標はVMC（Unity）の左手系（x右・y上・z前）。WebXRから位置(x, y, −z)、回転(−x, −y, z, w)。スティックのyはGamepad APIと逆（上が +）。
- `Trigger` / `Grip` / `Stick`の名前はこのプロジェクトの取り決めです（VMCにアナログ入力の決まった名前がないため）。
- ボーン・ブレンドシェイプは送りません（アバターのIKはサーバ側）。`/VMC/Ext/T`はfloat32なので、ページを長時間開いたままだと時刻の精度が落ちます（3時間で約1 ms）。
- 確認方法: Node（`--experimental-strip-types`）で`vmc.ts`を直接importしてバンドルを作り、`teleop.vmc.decode_sample`で読み戻して比較。aiortcのDataChannelを2つの`RTCPeerConnection`でループバックさせ、`PeerManager._on_message`に通すと実際の経路をIsaac Simなしで試せます。

### 没入表示

- Isaac Sim内のステレオカメラ（片目の正方形、視野角100°）がHMDに追従します。各フレームには描画に使った頭の姿勢を付け、`events`の`fmeta`で送ります。Web AppはRTPタイムスタンプでフレームと姿勢を対応付け、その姿勢の位置にワールド固定で表示します（`xr/frameMeta.ts`、`ImmersiveView.tsx`）。
- RGBアノテータは1回前のKit更新で描いた画像を返すので、付ける姿勢をこの遅れに合わせています（`IMMERSIVE_RENDER_LAG_FRAMES`）。ずれると首を振ったときに像が1フレーム分ずれます（`perf_check.py`のlagテストで確認）。
- 立体視（モニター表示）は表示中カメラの左右にIPD間隔のカメラを置き、左右並びで配信。Web Appが測ったIPDを`sendIpd`で送ります。

### アバター

- 読み込むVRM（VRM 0.x / 1.0）は`config.avatar_file()`が決めます。優先順は`--no-avatar`（なし）→ `--avatar FILE` → 環境変数`TELEOP_AVATAR` → `assets/avatars/avatar.vrm`（あるときだけ）。指定されたファイルがないときは既定のファイルに戻らず、警告を出してアバターなしで起動します（打ち間違いで別のアバターが出ないように）。相対パスは作業フォルダ、次にプロジェクトのルートから探します。`assets/avatars/`のほかのVRMは読みません。
- 起動時に`vrm.convert()`でUsdSkelへ変換し、`assets/avatars/_converted/<名前>/`に書き出します。出力の`.avatar.json`に記録したVRMの更新時刻と`CONVERTER_VERSION`が一致すれば変換を飛ばします。
- **姿勢**: 腰は首の下、背骨を曲げて首を頭に合わせ、腕は手首をコントローラに、指はトリガー（人差し指）とグリップ（中指〜小指）で曲げる。足は床に固定し、体が離れると踏み出す。大きさは頭の最高到達高さから求めた目の高さに合わせ、腕は最大1.3倍まで伸ばす。
- **Warp**: 骨格を扱う処理（IK、全身のFK、揺れもの94本、SkelAnimationに書く回転）はNVIDIA WarpのCPUカーネル（`avatar_ik.py` / `avatar_springs.py`）。IKは`SpringSolver.local`に直接書きます（コピーなし）。初回の起動時の停止を避けるため`wp.load_module`で先にロード。Pythonに残るのは骨格に依存しない目標の計算（体の向き、手・足の目標、足の踏み替え）。`config.py`の`AVATAR_IK_WARP` / `AVATAR_SPRINGS_WARP` / `AVATAR_FINGERS_BATCHED`をFalseでnumpyの参照実装（IKは`TELEOP_AVATAR_IK_WARP=0`でも）。更新は約19 ms → 約4 ms（IK約1、揺れもの 約1.2、書き込み 約1、当たり判定 約0.9 ms）。
- **揺れもの**: VRMのスプリングボーンと同じ計算（慣性・剛性・重力・空気抵抗、球の当たり判定）。表情（ブレンドシェイプ）は未対応。
- **見た目**: MToonをUsdPreviewSurfaceで近似。メインテクスチャを弱い拡散反射、影色テクスチャ（_ShadeTexture）を発光に使う。HDRの発光色の部分は自発光。係数は`vrm.py`の`TOON_LIT` / `TOON_SHADE` / `TOON_GLOW`。輪郭線とリムライトはなし。
- **一人称**: 表示中は没入表示のカメラの近クリップを20 cm（鏡像には影響しない）。
- **当たり判定**: 頭・胴2・上腕・前腕・手・太もも・すね・足の15個のキネマティック剛体をPhysXのkinematic targetで骨に追従させます。大きさは変換時にその骨に重み付けされた頂点から（揺れものは除外）。物を掴む手の手と前腕は当たり判定なし。無効時・非表示時は遠くに退避し、有効にした瞬間は瞬間移動。当たり判定は描画されません（RTX LiDARに見えるのはメッシュ。未検証）。

### 没入表示で物を掴む

グリップを押したとき、握り位置から6 cm以内に表面がある物体を掴み、離すと手の速さで放します。ロボットが掴んでいる物も取れます。手の役割は`set_teleop_settings`の`handMode`（`robot` / `grab`）。

### 映像の配信

- **コーデックの選択**: ブラウザはVP8を先に提示し、aiortcはオファーの順で選ぶため、`setRemoteDescription`の後に`nvenc.prefer_video_codec`がトランシーバの`_codecs`を並べ替えてH.264を優先します（RTXは元のコーデックの直後に保つ）。`setCodecPreferences`やSDPの書き換えでは効きませんでした。
- **PyAVのNVENC**（`nvenc.py`、接続ごとのエンコーダ）: `aiortc.codecs.H264Encoder`を差し替え。`h264_nvenc`、RGBAをそのまま、p1 / ull、VBR `cq 19` + maxrate/bufsize、baseline。受信側の帯域推定（REMB）は無視します（推定が低いまま上がらず、10%以上変わるたびにエンコーダを開き直していたため）。VBRでは`bit_rate`がNoneになるので開いたときの値を自分で保持。NVENCがなければlibx264。
- **GPUから直接**（`gpu_stream.py`、全接続で1本）: アノテータを`device="cuda"`で作り、`get_data(device="cuda")`のWarp配列をWarpカーネルで3枚のリングバッファの1枚に左右並びでコピー → `wp.record_event` → エンコーダスレッドが`synchronize_event`してからPyNvVideoCodecでエンコード（新しいフレームが来たら古い方を捨てる）。
  - `CreateEncoder(w, h, "ABGR", False, codec="h264", profile="baseline", preset="P1", tuning_info="ultra_low_latency", rc="vbr", …, bf=0, repeatspspps=1)`。入力は`.cuda()`が(H, W, 4) uint8のWarp配列を返すオブジェクト（配列そのもの・リストは不可）。
  - `Encode`のたびに`EndEncode()`を呼びます（呼ばないと出力が3フレーム遅れる）。キーフレームはpicflags `IDR(2) | SPSPPS(4)`。
  - NVENCはBT.709リミテッドレンジで変換するのに色情報を書かないので、PyAVの`BitStreamFilterContext("h264_metadata=…")`で書き込みます（入力ストリームが必要なので`av.open(BytesIO, "w", format="h264").add_stream("h264")`を使う）。
  - パケットは`FrameHub.publish_packet` → `SimVideoTrack.recv`が`av.Packet`を返し、aiortcはpacketizeだけします。キーフレームの要求: 新しい接続、キューのあふれ（その接続は次のキーフレームまで待つ）、PLI（aiortcの非公開`sender._RTCRtpSender__force_keyframe`を差し替えて受け取る）。
  - `device.synchronize`を使うとメインスレッドで約6 ms待つので、イベントで同期しています。
  - VP8で接続したブラウザと`/api/snapshot`のときだけCPUに読み戻します（`get_data(device="cpu")`）。
- **解像度変更**: `set_resolution`（`screen` / `stereo` / `immersive` 1024〜2048 / `bitrate` 1〜100 Mbpsか`"auto"` / `dlss`）。描画プロダクトを作り直します。ビットレートの自動は`--bitrate`（1280×720での値）を画素数に比例させ、上限60 Mbps。H.264の幅の上限から、没入の片目2048（配信4096×2048）が最大。
- **DLSS**: `/rtx/post/aa/op`（3 = DLSS、1 = TAA）と`/rtx/post/dlss/execMode`（0性能、1バランス、2画質、3自動）。Isaac Simの既定は性能（0）。
- **ビューポート**: Isaac Sim自身のビューポートは配信カメラと同じだけ描画コストがかかるため、`--viewport auto`では受信中とヘッドレス時に止めます（`get_active_viewport().updates_enabled`）。

### 計測

- サーバ側の時間は`perf.py`の`PERF`に集めます（受信中は`teleop.perf`に5秒ごと。Quest側の数値は`events`の`perf`で届く）。2Dパネルの「映像の計測」とVRのステータス行に表示。
- 項目: Kit更新（`physics.step`は物理1ステップ）、前処理 / アバター、画像取得 / 左右の連結（`grab.*`。CPU経路ではRGBAのまま並べ、アルファの除去とYUV変換はエンコーダ側のスレッド）、頭の姿勢反映 → 配信、エンコード（`enc.*`）、ブラウザ側のデコード / 受信バッファ待ち / サーバ配信 → 表示、頭の動き → 表示（没入表示の往復）。
- Web Appの表示までの時間は`perfStats.ts`（`getStats`と`requestVideoFrameCallback`）。QuestのWebXR中のrVFCは`expectedDisplayTime`とコールバックの`now`が`performance.now()`と別の時計（約100秒ずれる）なので、差だけを使います。サーバ配信 → 表示はpingで合わせた時計で計算。
- **Kitプロファイラ**: サーバを`--kit-trace <出力.json>`で起動し、`perf_check.py --kit-trace immersive`（`immersive-noavatar` / `mono` / `stereo`）で60フレーム記録 → サーバが止まってChromeトレース形式のJSON（150 MB程度）を書き出し → `kit_trace_report.py`。設定は`--/app/profilerBackend=cpu --/app/profileFromStart=true --/profiler/enabled=true --/plugins/carb.profiler-cpu.plugin/saveProfile=true …filePath=…`、トレースは正常終了時に書かれます。Pythonのゾーン名には` (Python)`が付き、キャプチャのマスクでは記録は止まらないので、レポートは最後のNフレームを使います。記録中は1〜2 fps遅くなります。

### 参考値（RTX 3090、ヘッドレス、物理30 Hz）

- シミュレーションのfps: モニター単眼 約27、立体視 約17〜21、没入 約21（アバター表示 約19）。配信はおおむね18〜20 fps。
- Kit更新の内訳（没入表示）: PhysX約23 ms（simulateの待ち12.7、Fabricの同期 約5、描画用の変換3.75）、hydraRender約15.6 ms（3ビュー）。ビューポート停止でKit更新−5〜7 ms（単眼・立体視・アバターなしの没入）。アバター表示時の没入では効果が小さい（p95のスパイクあり。未調査）。
- 没入の片目解像度ごと（GPUから直接、アバターあり）: 1024約22 fps、1536約18 fps、2048約14 fps（以前のCPU経路では1536約15、2048約11）。メインスレッドの画像取得は片目2048で約21 ms → 約2.5 ms。2048ではKitの描画（約58 ms）が支配的。
- DLSS（片目2048のKit更新）: 性能 約54 ms、画質 約72 ms、ネイティブ 約77 ms。
- エンコード（立体視1フレーム）: CPUのVP8約11 ms → PyAVのNVENC約4 ms。

## Web Appのメモ

- 設定のうちブラウザに保存するものは`store.ts`の`persist`の`partialize`に追加します（例: `swapEyes`、`screenDistance`）。
- 通常の2D UIは`components/ControlPanel.tsx`、VR内のメニューは`xr/VRPanel.tsx`。ビットレートのスライダーは、ドラッグ中に毎回送らないよう500 msのデバウンスで送ります（依存配列は`[draft]`だけにし、ストアは`useApp.getState()`で読む。ストアのオブジェクトを依存に入れると止まらなくなった）。
- 開発サーバ: `cd webapp && npm run dev`（https://PC:5173、`/api`を8443にプロキシ）。
