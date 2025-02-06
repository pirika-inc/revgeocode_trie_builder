import json

import math
import pickle
from collections import defaultdict

import click


def crossx(y: float, ax: float, ay: float, bx: float, by: float) -> float | None:
    """線分(ax,ay)-(bx,by)とy座標yの交点のx座標を求める
    :param y:
    :param ax:
    :param ay:
    :param bx:
    :param by:
    :return: 交点のx座標。存在しない場合はNone
    """
    if ay == by:
        return None
    if ay <= y <= by or by <= y <= ay:
        return (bx - ax) * (y - ay) / (by - ay) + ax
    return None


def expand(x: int) -> int:
    """xを2進数表記し、隣り合う桁の間に0を挿入する
    :param x:
    :return:
    """
    x = (x & 0xffff0000ffff0000) << 16 | x & 0x0000ffff0000ffff
    x = (x & 0xff00ff00ff00ff00) << 8 | x & 0x00ff00ff00ff00ff
    x = (x & 0xf0f0f0f0f0f0f0f0) << 4 | x & 0x0f0f0f0f0f0f0f0f
    x = (x & 0xcccccccccccccccc) << 2 | x & 0x3333333333333333
    x = (x & 0xaaaaaaaaaaaaaaaa) << 1 | x & 0x5555555555555555
    return x


def encode(x: int, y: int) -> int:
    """geocellの数値表現を返す。
    :param x: 経度に対応する。何番目の経度ユニットか
    :param y: 緯度に対応する。何番目の緯度ユニットか
    :return: geocellの数値表現
    """
    return expand(y) << 1 | expand(x)


def toco(x: float, y: float, resolution: int) -> tuple[int, int]:
    """緯度経度から(経度ユニット番号, 緯度ユニット番号)に変換する
    :param x: 経度
    :param y: 緯度
    :param resolution: geocellの解像度
    :return: (経度ユニット番号, 緯度ユニット番号)
    """
    yunit = (90 - (-90)) / (1 << resolution * 2)
    xunit = (180 - (-180)) / (1 << resolution * 2)
    return int((x - (-180)) / xunit), int((y - (-90)) / yunit)


def plot(css: list[list[list[float]]], resolution: int) -> list[int]:
    """中心点が多角形cssに含まれるようなgeocellのリストを返す
    :param css: 多角形のリスト
    :param resolution: geocellの解像度
    :return: 中心点が多角形のどれかに含まれるようなgeocellのリスト
    """
    miny, maxy = 90.0, -90.0
    for cs in css:
        for x, y in cs:
            miny = min(miny, y)
            maxy = max(maxy, y)
    yunit = (90 - (-90)) / (1 << resolution * 2)
    xunit = (180 - (-180)) / (1 << resolution * 2)

    minyi = math.floor((miny - (-90 + yunit / 2)) / yunit)
    maxyi = math.ceil((maxy - (-90 + yunit / 2)) / yunit)

    ret = []

    # 各y座標ユニットに対して、中心点のy座標を通っている辺の交点のx座標のリストを格納する
    xss = [[] for _ in range(minyi, maxyi + 1)]

    for cs in css:
        n = len(cs)
        for i in range(n):
            lminyi = math.floor((min(cs[i][1], cs[(i + 1) % n][1]) - (-90 + yunit / 2)) / yunit)
            lmaxyi = math.ceil((max(cs[i][1], cs[(i + 1) % n][1]) - (-90 + yunit / 2)) / yunit)
            for yi in range(lminyi, lmaxyi + 1):
                y = -90 + yunit / 2 + yunit * yi
                x = crossx(y, cs[i][0], cs[i][1], cs[(i + 1) % n][0], cs[(i + 1) % n][1])
                if x is not None:
                    xss[yi - minyi].append(x)

    # 各y座標ユニットで、0-indexedで2k番目と2k+1番目の交点の間にある点が、多角形に含まれる中心点
    for yi in range(minyi, maxyi + 1):
        xs = xss[yi - minyi]

        assert len(xs) % 2 == 0
        xs.sort()
        for i in range(0, len(xs), 2):
            minxi = math.ceil((xs[i] - (-180 + xunit / 2)) / xunit)
            maxxi = math.floor((xs[i + 1] - (-180 + xunit / 2)) / xunit)
            for xi in range(minxi, maxxi + 1):
                ret.append(encode(xi, yi))

    return ret


def cross(l: float, r: float, d: float, u: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    """線分(x1,y1)-(x2,y2)と矩形(l,d,r,u)が交点を持つかどうかを返す
    :param l: 矩形の左端
    :param r: 矩形の右端
    :param d: 矩形の下端
    :param u: 矩形の上端
    :param x1: 線分の端点1のx座標
    :param y1: 線分の端点1のy座標
    :param x2: 線分の端点2のx座標
    :param y2: 線分の端点2のy座標
    :return: 交点を持つかどうか
    """
    assert l <= r
    assert d <= u
    if x1 == x2:
        return l <= x1 <= r and min(y1, y2) <= u and max(y1, y2) >= d
    if y1 == y2:
        return d <= y1 <= u and min(x1, x2) <= r and max(x1, x2) >= l
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    assert a != 0
    if a > 0:
        return a * l + b <= u and a * r + b >= d
    else:
        return a * l + b >= d and a * r + b <= u


def plot_edge(css: list[list[list[float]]], resolution: int) -> set[int]:
    """多角形の辺を含むgeocellの集合を返す
    :param css: 多角形のリスト
    :param resolution: geocellの解像度
    :return: 辺が多角形のどれかに含まれるようなgeocellの集合
    """
    yunit = (90 - (-90)) / (1 << resolution * 2)
    xunit = (180 - (-180)) / (1 << resolution * 2)

    ret = set()

    for cs in css:
        assert cs[0] == cs[-1]
        for i in range(len(cs) - 1):
            x1, y1 = cs[i]
            x2, y2 = cs[i + 1]

            cur = [*toco(x1, y1, resolution)]
            tar = [*toco(x2, y2, resolution)]
            ret.add(encode(*cur))
            while cur != tar:
                if cur[0] != tar[0]:
                    dx = 1 if cur[0] < tar[0] else -1
                    if cross(-180 + xunit * (cur[0] + dx), -180 + xunit * (cur[0] + dx + 1), -90 + yunit * cur[1],
                             -90 + yunit * (cur[1] + 1), x1, y1, x2, y2):
                        ret.add(encode(*cur))
                        cur[0] += dx
                if cur[1] != tar[1]:
                    dy = 1 if cur[1] < tar[1] else -1
                    if cross(-180 + xunit * cur[0], -180 + xunit * (cur[0] + 1), -90 + yunit * (cur[1] + dy),
                             -90 + yunit * (cur[1] + dy + 1), x1, y1, x2, y2):
                        ret.add(encode(*cur))
                        cur[1] += dy
    return ret


def lift_unique_edges(trie: list[list[int] | str], cities: list[str], unit: int) -> list[list[int] | str]:
    """trieで、ノードの辺の出る先が1通り(-1を除く)しかない場合、そのノードからすべてその先に行くとして、そのノードを削除して縮約する
    :param trie: trie
    :param cities: 市区町村のリスト
    :param unit:
    :return: 縮約後のtrie
    """
    # prune
    parents = [None] * len(trie)  # parents[x] = (ノードxに至る辺の元ノード, 辺番号)
    for i in range(len(trie)):
        if type(trie[i]) is str:
            continue
        for j in range(1 << unit):
            if trie[i][j] != -1 and trie[i][j] >= len(cities) + 1:
                parents[trie[i][j]] = (i, j)

    deleted = [False] * len(trie)
    for i in range(len(trie) - 1, len(cities), -1):
        unique_edge = -1
        for j in trie[i]:
            if j != -1 and j != unique_edge:
                if unique_edge == -1:
                    unique_edge = j
                else:
                    unique_edge = -1  # 2回目の来訪は失敗とする
                    break
        if unique_edge != -1:
            assert parents[i] is not None
            trie[parents[i][0]][parents[i][1]] = unique_edge
            deleted[i] = True

    maps = [-1] * len(trie)
    p = 0
    for i in range(len(trie)):
        if not deleted[i]:
            maps[i] = p
            p += 1

    # 番号を振り直す
    ntrie = []
    for i in range(len(trie)):
        if deleted[i]:
            continue
        row = trie[i]
        if i >= len(cities) + 1:
            for j in range(1 << unit):
                if row[j] != -1:
                    row[j] = maps[row[j]]
        ntrie.append(row)
    return ntrie


def shrink(points: dict[str, list[int]], unit: int, resolution: int) -> dict[str, list[tuple[int, int]]]:
    """pointsにはちょうどresolutionの解像度のgeocellしか入っていない。これに対し、1<<unit進数で末尾桁以外同じgeocellが全て同じ市区町村をさしているなら、末尾桁を削除する
    :param points:
    :param unit:
    :param resolution:
    :return: 縮約後のpoints. {市区町村: (geocell, geocellの解像度)のリスト}
    """
    lim = resolution * 4 // unit

    npoints = {}
    for k, v in points.items():
        print(k)
        lals: list[set[int] | None] = [None] * (lim + 1)
        lals[lim] = set(v)
        for i in range(lim - 1, -1, -1):
            nlal = set()
            for w in lals[i + 1]:
                # geocellの末尾桁が0のとき、末尾桁以外が同じすべてがlals[i+1]に入っていればOK.
                if (w & (1 << unit) - 1) == 0 and all(w + j in lals[i + 1] for j in range(1 << unit)):
                    nlal.add(w >> unit)
            # i+1層の不要になったgeocellを削除
            for w in nlal:
                for j in range(1 << unit):
                    lals[i + 1].remove(w << unit | j)
            lals[i] = nlal

        lst = []
        for i in range(len(lals)):
            for w in lals[i]:
                lst.append((w, i))
        npoints[k] = lst

    return npoints


def build_trie(points: dict[str, list[tuple[int, int]]], unit: int) -> list[list[int] | str]:
    """pointsを元にtrieを構築する
    :param points:
    :param unit:
    :return:
    """

    cities = list(points.keys())

    city_index = {city: i for i, city in enumerate(cities)}

    trie = [[-1] * (1 << unit)]
    trie.extend(cities)

    for k, v in points.items():
        print(k)
        for w, le in v:
            cur = 0
            for j in range(le):
                x = w >> unit * (le - 1 - j) & ((1 << unit) - 1)
                ne = trie[cur][x]
                if ne == -1:
                    if j == le - 1:
                        trie[cur][x] = city_index[k] + 1
                    else:
                        trie.append([-1] * (1 << unit))
                        ne = len(trie) - 1
                        trie[cur][x] = ne
                    cur = trie[cur][x]
                else:
                    cur = ne

    print("original", len(trie))

    # 海上に大きくはみ出すことがある。通常の運用では問題ないが、
    # 位置情報を海上に設定して投稿し、それが見える化ページに表示されうるのが問題とのことでコメントアウト
    # trie = lift_unique_edges(trie, cities, unit)
    # print("lift", len(trie))

    trie = identify_same_rows(trie)
    print("identify", len(trie))

    trie = shrink_by_pattern(trie)
    print("shrink", len(trie))

    return trie


def shrink_node(a: list[int]) -> list[int]:
    """ノードaを圧縮する。
    ノードから出る辺の先の種類数をkとすると、kを2進数で表現するのに必要なビット数をbとする。
    ノードの内容は、先頭要素以外は出る辺の先のユニークなリスト、先頭要素は各辺の行き先をbビットで表したものをまとめた整数になる。
    先頭要素は、16進ですべての要素が異なっていても64bitに収まる。
    :param a:
    :return: [ptn, *table]
    """
    table = sorted(list(set(a)))
    nk = len(table)
    bit = 1
    lnk = 2
    while nk > lnk:
        bit += 1
        lnk *= 2

    ptn = 0
    for i in range(len(a)):
        ptn |= table.index(a[i]) << bit * i

    # return [bit, ptn, *table]
    return [ptn, *table]


def shrink_by_pattern(trie: list[list[int] | str]) -> list[list[int] | str]:
    """trieの各ノードの内容を圧縮する
    :param trie:
    :return:
    """
    return [shrink_node(row) if type(row) is list else row for row in trie]


def identify_same_rows(trie: list[list[int] | str]) -> list[list[int] | str]:
    """trieで、同一内容のノードを同一視して縮約する
    :param trie:
    :return:
    """
    tos = [-1] * len(trie)
    data_index = {}  # data_index[data] = index
    for i, row in enumerate(trie):
        idx = tuple(row) if type(row) is list else row
        tos[i] = data_index.setdefault(idx, len(data_index))

    ntrie = [[]] * len(data_index)
    for k, idx in data_index.items():
        ntrie[idx] = list(k) if type(k) is tuple else k
        if type(trie[idx]) is list:
            ntrie[idx] = [tos[v] if v != -1 else -1 for v in ntrie[idx]]
    return ntrie


def process_feature(feature: dict, resolution: int) -> tuple[str, list[int], set[int]] | None:
    pref = feature["properties"]["N03_001"] or ""
    county = feature["properties"]["N03_003"] or ""
    city = feature["properties"]["N03_004"] or ""
    ku = feature["properties"]["N03_005"] or ""
    if city == "所属未定地":
        return None

    code = pref + "/" + county + city + ku
    res_plot = plot(feature["geometry"]["coordinates"], resolution)
    res_edge = plot_edge(feature["geometry"]["coordinates"], resolution)
    return code, res_plot, res_edge


@click.command()
@click.argument('geojsonpath', type=click.Path(exists=True), required=True)
@click.option('--unit', required=False, default='4', type=click.Choice(['1', '2', '4']))
@click.option('--resolution', type=int, required=False, default=9)
@click.option('--outjsonpath', required=False)
@click.option('--outpicklepath', required=False)
@click.option('--multiprocess', is_flag=True, default=False)
def main(geojsonpath, unit, resolution, outjsonpath, outpicklepath, multiprocess):
    """
    :param geojsonpath: 行政区域を表すgeojsonファイルのパス
    :param unit: 出力するgeocellをまとめる粒度の単位。同一市町村のgeocellをまとめるとき、1,2,4に対してそれぞれ2進数, 4進数, 16進数で同一市町村なら1桁削ることを行う。
    :param resolution: 16進の場合のgeocellの解像度。
    :param outjsonpath: 出力するjsonファイルのパス
    :param outpicklepath: 出力するpickleファイルのパス
    :param multiprocess: 並列処理を行うかどうか
    :return:
    """
    unit = int(unit)

    obj = json.load(open(geojsonpath, "rb"))

    points = defaultdict(list)
    al = 0
    edges = {}
    already = set()

    if multiprocess:
        from multiprocessing import Pool
        from functools import partial
        with Pool() as p:
            for r in p.imap_unordered(partial(process_feature, resolution=resolution), obj["features"]):
                if r is None:
                    continue
                code, res_plot, res_edge = r
                al += len(res_plot)
                already |= set(res_plot)

                for w in res_edge:
                    edges[w] = code if w not in edges else None
                points[code].extend(res_plot)
                print(code, "center", al, "edges", len(edges))
    else:
        for feature in obj["features"]:
            res = process_feature(feature, resolution)
            if res is None:
                continue
            code, res_plot, res_edge = res
            al += len(res_plot)
            already |= set(res_plot)

            for w in res_edge:
                edges[w] = code if w not in edges else None
            points[code].extend(res_plot)
            print(code, "center", al, "edges", len(edges))

    for k, v in edges.items():
        if k not in already and v is not None:
            points[v].append(k)

    points = shrink(points, unit, resolution)

    print("build trie")
    trie = build_trie(points, unit)

    if outjsonpath:
        json.dump(trie, open(outjsonpath, "w"), separators=(",", ":"), ensure_ascii=False)

    if outpicklepath:
        pickle.dump(trie, open(outpicklepath, "wb"), protocol=4)


if __name__ == "__main__":
    main()
