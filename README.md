# num-cpp

学習用のC++数値計算ライブラリ

## 主な機能

### `num/core/vector.hpp`

- ベクトル、ゼロベクトルの生成
- 基本演算（加減算、符号反転、スカラー乗除算）
- ベクトル演算（内積）
- L1、L2、無限大ノルム

### `num/core/matrix.hpp`

- 行列、ゼロ行列、単位行列の生成
- 基本演算（加減算、符号反転、スカラー乗除算）
- 行列演算（積、転置）
- L1、無限大、フロベニウスノルム

### `num/linalg/lu.hpp`

- LU分解（`PA = LU`）
- LU分解を用いた連立一次方程式の求解、行列式、逆行列の計算

## 対応範囲

### ベクトル、行列

- 動的サイズの密ベクトル、密行列のみ
- 要素型は浮動小数点型のみ

### LU分解

- 正方行列のみ
- 部分ピボット選択のみ
- 特異性はピボットが厳密に0と等しいか否かで判定する

## 動作環境

- C++23 対応コンパイラ
- CMake 3.25 以降
- サニタイザーとカバレッジは Clang のみ対応

## ビルドとテスト

```bash
# 例: Clang
cmake -S . -B build/clang-debug -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON
cmake --build build/clang-debug
ctest --test-dir build/clang-debug --output-on-failure
```

```bash
# 例: Sanitizers
cmake -S . -B build/clang-sanitizers -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON \
  -DNUM_ENABLE_SANITIZERS=ON
cmake --build build/clang-sanitizers
ctest --test-dir build/clang-sanitizers --output-on-failure
```

```bash
# 例: Coverage
cmake -S . -B build/clang-coverage -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON \
  -DNUM_ENABLE_COVERAGE=ON
cmake --build build/clang-coverage --target coverage
```

## ビルドオプション

| オプション名            | 説明                                      | 初期値 |
| ----------------------- | ----------------------------------------- | :----: |
| `BUILD_TESTING`         | テストのビルド                            | `OFF`  |
| `NUM_BUILD_EXAMPLES`    | サンプルのビルド                          | `OFF`  |
| `NUM_ENABLE_SANITIZERS` | ASan + UBSan を有効化（Clangのみ）        | `OFF`  |
| `NUM_ENABLE_COVERAGE`   | コードカバレッジ計測を有効化（Clangのみ） | `OFF`  |
