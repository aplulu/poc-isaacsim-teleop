# 掴む対象モデルの追加

`assets/props/<セット>/`に3Dモデルを置いて`C:\isaacsim\python.bat server\build_scene.py`を実行すると、
そのセットの掴む対象としてシーンに追加されます。`<セット>`は`fruit`（果物）/ `part`（部品）/ `tool`（工具）/ `electrical`（電装品）/ `fastener`（ボルト・ナット）/ `smallpart`（小部品）/ `small`（小物）です。

- 対応形式: `.usd` `.usda` `.usdc` `.usdz`（そのまま）、`.glb` `.gltf` `.obj` `.fbx`（`assets/_converted/`にUSD変換）
- ファイル名が対象名になり、英語の指示文に使われます（例: `small/stapler.glb` → "put the stapler in the tray"）。
- 大きさは一番長い辺が7 cmになるよう自動で拡大縮小されます（`server/teleop/config.py`の`CUSTOM_PROP_SIZE`。2F-85の開口は85 mm）。
- 剛体（質量100 g、凸包の当たり判定）として扱い、Y-upのモデルは自動でZ-upに起こします。
- 置き場所は`CUSTOM_PROP_SLOTS`の順に割り当てられます。
