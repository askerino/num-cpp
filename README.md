# num-cpp

[![CI](https://github.com/askerino/num-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/askerino/num-cpp/actions/workflows/ci.yml)
[![Docs](https://github.com/askerino/num-cpp/actions/workflows/docs.yml/badge.svg)](https://github.com/askerino/num-cpp/actions/workflows/docs.yml)

学習用のC++数値計算ライブラリです。

C++と数値計算の基礎を学ぶために作っています。実際の開発ではEigenなどの既存ライブラリを使う前提で、内部の仕組みを理解することを目的としています。外部ライブラリに頼らず、数式との対応を意識して実装しています。

## 主な機能

ヘッダーオンリーライブラリです。基本データ構造は`num`、線形代数アルゴリズムは`num::linalg`名前空間にあります。

### ベクトル（`num/core/vector.hpp`）

- ベクトル、ゼロベクトルの生成
- 基本演算（加減算、符号反転、スカラー乗除算）
- ベクトル演算（内積）
- L1ノルム、L2ノルム、無限大ノルム

### 行列（`num/core/matrix.hpp`）

- 行列、ゼロ行列、単位行列の生成
- 基本演算（加減算、符号反転、スカラー乗除算）
- 行列演算（積、転置）
- 誘導L1ノルム、誘導無限大ノルム、フロベニウスノルム

### LU分解（`num/linalg/lu.hpp`）

- LU分解（`PA = LU`）
- LU分解を用いた連立一次方程式の求解、行列式、逆行列の計算

## 制約事項

### ベクトル、行列

- 動的サイズの密ベクトル、密行列のみ
- 要素型は浮動小数点型のみ

### LU分解

- 正方行列のみ
- 部分ピボット選択のみ
- ピボットの絶対値が厳密に0の場合のみ特異と判定

## エラー処理

- APIの使用方法が誤っている場合（行列のサイズ不一致など）は例外を投げる
- 正しい入力でも発生し得る数値計算上のエラーは`std::expected<T, num::Error>`で返す
- `num::Error`のエラーメッセージは`num::error_message`で取得する

| `num::Error`       | 原因                   |
| ------------------ | ---------------------- |
| `singular_matrix`  | 特異行列               |
| `non_finite_input` | NaNまたはInfを含む入力 |

## 動作環境

- C++23対応コンパイラ
- CMake 3.25以降

## 使い方

`num::num`ターゲットをリンクして使用します。

```cmake
include(FetchContent)
FetchContent_Declare(
        num-cpp
        GIT_REPOSITORY https://github.com/askerino/num-cpp.git
        GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(num-cpp)

target_link_libraries(app PRIVATE num::num)
```

```cpp
#include "num/linalg/lu.hpp"

#include <cstdio> 
#include <print>

int main() {
    const num::Matrix<double> a{
        {2.0, 1.0, -1.0},
        {1.0, 3.0, 2.0},
        {3.0, -1.0, 1.0},
    };
    const num::Vector<double> b{8.0, 6.0, 5.0};

    const auto x = num::linalg::lu_solve(a, b);
    if (!x) {
        std::println(stderr, "error: {}", num::error_message(x.error()));
        return 1;
    }

    std::println("residual: {:.3e}", (a * (*x) - b).norm_l2());
}
```

詳しいサンプルコードは[`examples/solve_linear_system.cpp`][example]を参照してください。

[example]: https://github.com/askerino/num-cpp/blob/main/examples/solve_linear_system.cpp

## APIドキュメント

公開APIのドキュメントは、ヘッダーファイル内のDoxygenコメントとして記述しています。

オンライン版は<https://askerino.github.io/num-cpp/>で閲覧できます。

## 開発

### ディレクトリ構成

```text
.
├── .github/workflows/
├── cmake/
├── examples/
│   ├── CMakeLists.txt
│   └── solve_linear_system.cpp
├── include/
│   └── num/
│       ├── core/
│       └── linalg/
├── tests/
│   ├── core/
│   ├── linalg/
│   └── CMakeLists.txt
└── CMakeLists.txt
```

### 必要なツール

- Git
- Ninja（任意、ビルド例で使用）
- カバレッジは使用するClangと同じバージョンの`llvm-profdata`と`llvm-cov`が必要
- フォーマットは`clang-format` 21、静的解析は`clang-tidy` 21が必要
- APIドキュメントの生成はDoxygen 1.9.5以降とGraphvizが必要

### ビルド

```bash
cmake -S . -B build/clang-debug -G Ninja \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug
cmake --build build/clang-debug
```

サンプルは`build/clang-debug/examples/solve_linear_system`で実行できます。

#### ビルドオプション

| オプション名            | 説明                                          | 初期値 |
| ----------------------- | --------------------------------------------- | :----: |
| `NUM_BUILD_TESTS`       | テストをビルドする                            | `ON`※  |
| `NUM_BUILD_EXAMPLES`    | サンプルをビルドする                          | `ON`※  |
| `NUM_ENABLE_SANITIZERS` | ASan + UBSanを有効にする（Clangのみ）         | `OFF`  |
| `NUM_ENABLE_COVERAGE`   | コードカバレッジ計測を有効にする（Clangのみ） | `OFF`  |
| `NUM_BUILD_DOCS`        | APIドキュメントを生成する（Doxygen）          | `OFF`  |

※他のプロジェクトから`FetchContent`などで取り込んだ場合、初期値は`OFF`になります。

### テスト

ビルド完了後、次のコマンドで全テストを実行します。

```bash
ctest --test-dir build/clang-debug --output-on-failure
```

#### サニタイザー付きテスト

```bash
cmake -S . -B build/clang-sanitizers -G Ninja \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DNUM_ENABLE_SANITIZERS=ON
cmake --build build/clang-sanitizers
ctest --test-dir build/clang-sanitizers --output-on-failure
```

#### カバレッジ

```bash
cmake -S . -B build/clang-coverage -G Ninja \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DNUM_ENABLE_COVERAGE=ON \
    -DNUM_BUILD_EXAMPLES=OFF
cmake --build build/clang-coverage --target coverage
```

サニタイザーとカバレッジは同時に有効にできません。

### フォーマット

`clang-format` 21を使用します。設定は`.clang-format`を参照してください。

### 静的解析

`clang-tidy` 21を使用します。設定は`.clang-tidy`と`tests/.clang-tidy`を参照してください。

### APIドキュメントの生成

```bash
cmake -S . -B build/docs \
    -DNUM_BUILD_TESTS=OFF \
    -DNUM_BUILD_EXAMPLES=OFF \
    -DNUM_BUILD_DOCS=ON
cmake --build build/docs --target docs
```

`build/docs/docs/html/index.html`をブラウザで開くと閲覧できます。

## 実装予定

- 基本的な統計量

## ライセンス

[MIT License](LICENSE)
