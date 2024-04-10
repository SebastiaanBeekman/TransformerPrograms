import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/dyck2/trainlength30/s1/dyck2_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {
            0,
            6,
            8,
            30,
            32,
            35,
            38,
            40,
            41,
            42,
            43,
            44,
            45,
            48,
            50,
            52,
            53,
            54,
            55,
            56,
            58,
            59,
        }:
            return k_position == 5
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {57, 39, 5, 31}:
            return k_position == 4
        elif q_position in {33, 34, 36, 37, 7, 9, 11, 46, 47, 49, 51}:
            return k_position == 6
        elif q_position in {10, 27, 29}:
            return k_position == 7
        elif q_position in {12, 14}:
            return k_position == 10
        elif q_position in {13}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 11
        elif q_position in {16, 18}:
            return k_position == 14
        elif q_position in {17, 19}:
            return k_position == 15
        elif q_position in {20}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 19
        elif q_position in {22}:
            return k_position == 20
        elif q_position in {23}:
            return k_position == 21
        elif q_position in {24, 26}:
            return k_position == 22
        elif q_position in {25}:
            return k_position == 8
        elif q_position in {28}:
            return k_position == 26

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 42, 35, 58}:
            return k_position == 30
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 23
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4, 6}:
            return k_position == 3
        elif q_position in {5, 31}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9, 11}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {12, 13}:
            return k_position == 9
        elif q_position in {14, 15}:
            return k_position == 10
        elif q_position in {16, 18, 21, 25, 29}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20, 22}:
            return k_position == 19
        elif q_position in {27, 23}:
            return k_position == 14
        elif q_position in {24, 26, 28}:
            return k_position == 21
        elif q_position in {49, 50, 30}:
            return k_position == 49
        elif q_position in {32}:
            return k_position == 56
        elif q_position in {33, 34, 57}:
            return k_position == 36
        elif q_position in {51, 36}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 42
        elif q_position in {56, 38}:
            return k_position == 50
        elif q_position in {39}:
            return k_position == 59
        elif q_position in {40, 53}:
            return k_position == 58
        elif q_position in {41}:
            return k_position == 40
        elif q_position in {43}:
            return k_position == 54
        elif q_position in {44}:
            return k_position == 35
        elif q_position in {45, 46}:
            return k_position == 29
        elif q_position in {47}:
            return k_position == 45
        elif q_position in {48}:
            return k_position == 33
        elif q_position in {52}:
            return k_position == 55
        elif q_position in {54}:
            return k_position == 38
        elif q_position in {55}:
            return k_position == 39
        elif q_position in {59}:
            return k_position == 37

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"{", "(", ")"}:
            return k_token == ")"
        elif q_token in {"<s>"}:
            return k_token == ""
        elif q_token in {"}"}:
            return k_token == "}"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 53
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {57, 2}:
            return k_position == 2
        elif q_position in {3, 4}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 5
        elif q_position in {43, 7}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9, 11}:
            return k_position == 7
        elif q_position in {10, 27, 29}:
            return k_position == 8
        elif q_position in {12, 14}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 12
        elif q_position in {16, 18}:
            return k_position == 15
        elif q_position in {17, 19}:
            return k_position == 16
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24, 26}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 9
        elif q_position in {51, 28}:
            return k_position == 27
        elif q_position in {30}:
            return k_position == 35
        elif q_position in {34, 31}:
            return k_position == 34
        elif q_position in {32}:
            return k_position == 29
        elif q_position in {33, 55}:
            return k_position == 47
        elif q_position in {35}:
            return k_position == 55
        elif q_position in {58, 36}:
            return k_position == 46
        elif q_position in {37}:
            return k_position == 31
        elif q_position in {38}:
            return k_position == 48
        elif q_position in {48, 59, 39}:
            return k_position == 30
        elif q_position in {40, 52}:
            return k_position == 45
        elif q_position in {41}:
            return k_position == 52
        elif q_position in {42, 47}:
            return k_position == 40
        elif q_position in {44}:
            return k_position == 28
        elif q_position in {45}:
            return k_position == 56
        elif q_position in {56, 50, 46}:
            return k_position == 38
        elif q_position in {49}:
            return k_position == 59
        elif q_position in {53}:
            return k_position == 58
        elif q_position in {54}:
            return k_position == 49

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 25
        elif q_position in {2}:
            return k_position == 13
        elif q_position in {25, 3, 12}:
            return k_position == 27
        elif q_position in {4, 11, 45, 47, 55}:
            return k_position == 45
        elif q_position in {5}:
            return k_position == 33
        elif q_position in {27, 6}:
            return k_position == 50
        elif q_position in {37, 7, 51, 52, 54}:
            return k_position == 0
        elif q_position in {8, 16, 56}:
            return k_position == 48
        elif q_position in {9}:
            return k_position == 44
        elif q_position in {10}:
            return k_position == 53
        elif q_position in {49, 13}:
            return k_position == 35
        elif q_position in {32, 35, 14, 50, 31}:
            return k_position == 29
        elif q_position in {36, 15}:
            return k_position == 36
        elif q_position in {17}:
            return k_position == 46
        elif q_position in {18}:
            return k_position == 31
        elif q_position in {19, 28}:
            return k_position == 43
        elif q_position in {20}:
            return k_position == 51
        elif q_position in {21, 46}:
            return k_position == 57
        elif q_position in {57, 53, 22}:
            return k_position == 52
        elif q_position in {48, 34, 23}:
            return k_position == 34
        elif q_position in {24, 38}:
            return k_position == 54
        elif q_position in {26, 39}:
            return k_position == 42
        elif q_position in {29}:
            return k_position == 10
        elif q_position in {59, 30}:
            return k_position == 56
        elif q_position in {33}:
            return k_position == 41
        elif q_position in {40}:
            return k_position == 9
        elif q_position in {41}:
            return k_position == 6
        elif q_position in {42}:
            return k_position == 4
        elif q_position in {43}:
            return k_position == 30
        elif q_position in {44}:
            return k_position == 1
        elif q_position in {58}:
            return k_position == 37

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"("}:
            return position == 29
        elif token in {")"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 58
        elif token in {"{"}:
            return position == 28
        elif token in {"}"}:
            return position == 8

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 46}:
            return k_position == 58
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2}:
            return k_position == 24
        elif q_position in {32, 33, 3, 37, 7, 48, 49, 53, 30}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 17
        elif q_position in {5}:
            return k_position == 26
        elif q_position in {6}:
            return k_position == 51
        elif q_position in {8, 9}:
            return k_position == 55
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 31
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 28
        elif q_position in {15}:
            return k_position == 33
        elif q_position in {16}:
            return k_position == 53
        elif q_position in {17}:
            return k_position == 56
        elif q_position in {18, 28, 22}:
            return k_position == 38
        elif q_position in {34, 19}:
            return k_position == 35
        elif q_position in {42, 20}:
            return k_position == 57
        elif q_position in {21, 23}:
            return k_position == 22
        elif q_position in {24, 39}:
            return k_position == 44
        elif q_position in {25}:
            return k_position == 10
        elif q_position in {26, 58}:
            return k_position == 46
        elif q_position in {27, 52}:
            return k_position == 0
        elif q_position in {29}:
            return k_position == 16
        elif q_position in {56, 31}:
            return k_position == 39
        elif q_position in {35, 55}:
            return k_position == 5
        elif q_position in {59, 36, 44}:
            return k_position == 4
        elif q_position in {38, 40, 45, 51, 57}:
            return k_position == 1
        elif q_position in {41}:
            return k_position == 6
        elif q_position in {43}:
            return k_position == 41
        elif q_position in {50, 54, 47}:
            return k_position == 3

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 39
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 0
        elif q_position in {5}:
            return k_position == 28
        elif q_position in {35, 38, 6, 7, 44}:
            return k_position == 4
        elif q_position in {8, 9, 11}:
            return k_position == 3
        elif q_position in {10}:
            return k_position == 21
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13, 39}:
            return k_position == 51
        elif q_position in {14}:
            return k_position == 37
        elif q_position in {15}:
            return k_position == 20
        elif q_position in {16}:
            return k_position == 41
        elif q_position in {17, 37}:
            return k_position == 45
        elif q_position in {18}:
            return k_position == 34
        elif q_position in {19}:
            return k_position == 14
        elif q_position in {20}:
            return k_position == 26
        elif q_position in {26, 21}:
            return k_position == 30
        elif q_position in {42, 36, 22}:
            return k_position == 58
        elif q_position in {23}:
            return k_position == 47
        elif q_position in {24}:
            return k_position == 52
        elif q_position in {25}:
            return k_position == 31
        elif q_position in {50, 27}:
            return k_position == 33
        elif q_position in {28}:
            return k_position == 35
        elif q_position in {29}:
            return k_position == 55
        elif q_position in {51, 30}:
            return k_position == 53
        elif q_position in {49, 31}:
            return k_position == 7
        elif q_position in {32}:
            return k_position == 43
        elif q_position in {33, 46}:
            return k_position == 42
        elif q_position in {34, 52}:
            return k_position == 57
        elif q_position in {40, 53, 47}:
            return k_position == 8
        elif q_position in {41}:
            return k_position == 49
        elif q_position in {43}:
            return k_position == 54
        elif q_position in {56, 45, 54, 55}:
            return k_position == 1
        elif q_position in {48}:
            return k_position == 50
        elif q_position in {57}:
            return k_position == 40
        elif q_position in {58, 59}:
            return k_position == 48

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 4
        elif key in {("(", "}"), ("}", "(")}:
            return 22
        elif key in {(")", "("), (")", "<s>")}:
            return 12
        elif key in {("(", ")")}:
            return 19
        return 7

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, attn_0_0_output):
        key = (attn_0_3_output, attn_0_0_output)
        if key in {
            ("(", "("),
            ("(", "{"),
            ("<s>", "("),
            ("<s>", "{"),
            ("{", "("),
            ("{", "{"),
        }:
            return 4
        elif key in {
            (")", ")"),
            (")", "<s>"),
            (")", "}"),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 2
        elif key in {("}", "(")}:
            return 59
        elif key in {(")", "{")}:
            return 31
        return 39

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_0_output, attn_0_2_output):
        key = (attn_0_0_output, attn_0_2_output)
        if key in {("}", "("), ("}", "<s>"), ("}", "}")}:
            return 27
        elif key in {("(", "}")}:
            return 38
        elif key in {("(", "<s>")}:
            return 4
        elif key in {("(", "("), ("(", ")"), ("}", ")")}:
            return 24
        return 5

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_2_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_1_output, attn_0_2_output):
        key = (attn_0_1_output, attn_0_2_output)
        if key in {("<s>", "("), ("<s>", "<s>"), ("<s>", "{")}:
            return 51
        elif key in {
            ("(", "}"),
            (")", "}"),
            ("<s>", "}"),
            ("}", "("),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 42
        elif key in {("{", "}")}:
            return 33
        return 24

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_2_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_1_output):
        key = (num_attn_0_2_output, num_attn_0_1_output)
        return 50

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_1_output):
        key = (num_attn_0_0_output, num_attn_0_1_output)
        return 48

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 51

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 41

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_token, k_token):
        if q_token in {"{", "}", "(", ")"}:
            return k_token == "}"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_1_0_pattern = select_closest(tokens, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"{", "("}:
            return position == 1
        elif token in {"}", ")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 5

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_1_output, position):
        if attn_0_1_output in {"("}:
            return position == 4
        elif attn_0_1_output in {")"}:
            return position == 5
        elif attn_0_1_output in {"<s>"}:
            return position == 1
        elif attn_0_1_output in {"{"}:
            return position == 6
        elif attn_0_1_output in {"}"}:
            return position == 3

    attn_1_2_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_3_output, mlp_0_0_output):
        if attn_0_3_output in {"(", ")"}:
            return mlp_0_0_output == 7
        elif attn_0_3_output in {"<s>"}:
            return mlp_0_0_output == 5
        elif attn_0_3_output in {"{", "}"}:
            return mlp_0_0_output == 6

    attn_1_3_pattern = select_closest(mlp_0_0_outputs, attn_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_2_output, mlp_0_1_output):
        if attn_0_2_output in {"{", "("}:
            return mlp_0_1_output == 56
        elif attn_0_2_output in {"<s>", "}", ")"}:
            return mlp_0_1_output == 39

    num_attn_1_0_pattern = select(mlp_0_1_outputs, attn_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_2_output, mlp_0_0_output):
        if attn_0_2_output in {"("}:
            return mlp_0_0_output == 20
        elif attn_0_2_output in {"<s>", "}", ")"}:
            return mlp_0_0_output == 4
        elif attn_0_2_output in {"{"}:
            return mlp_0_0_output == 32

    num_attn_1_1_pattern = select(mlp_0_0_outputs, attn_0_2_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"<s>", "{", "("}:
            return mlp_0_0_output == 32
        elif attn_0_1_output in {"}", ")"}:
            return mlp_0_0_output == 4

    num_attn_1_2_pattern = select(mlp_0_0_outputs, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 33, 20, 53}:
            return k_mlp_0_0_output == 7
        elif q_mlp_0_0_output in {1, 10}:
            return k_mlp_0_0_output == 45
        elif q_mlp_0_0_output in {2}:
            return k_mlp_0_0_output == 1
        elif q_mlp_0_0_output in {3, 6, 7, 8, 9, 12, 50}:
            return k_mlp_0_0_output == 32
        elif q_mlp_0_0_output in {4}:
            return k_mlp_0_0_output == 10
        elif q_mlp_0_0_output in {5, 31}:
            return k_mlp_0_0_output == 39
        elif q_mlp_0_0_output in {59, 42, 11}:
            return k_mlp_0_0_output == 9
        elif q_mlp_0_0_output in {13}:
            return k_mlp_0_0_output == 4
        elif q_mlp_0_0_output in {37, 21, 14}:
            return k_mlp_0_0_output == 43
        elif q_mlp_0_0_output in {16, 15}:
            return k_mlp_0_0_output == 12
        elif q_mlp_0_0_output in {17}:
            return k_mlp_0_0_output == 37
        elif q_mlp_0_0_output in {18, 43}:
            return k_mlp_0_0_output == 54
        elif q_mlp_0_0_output in {19}:
            return k_mlp_0_0_output == 48
        elif q_mlp_0_0_output in {22, 47}:
            return k_mlp_0_0_output == 21
        elif q_mlp_0_0_output in {38, 49, 52, 23, 25}:
            return k_mlp_0_0_output == 51
        elif q_mlp_0_0_output in {24, 28}:
            return k_mlp_0_0_output == 14
        elif q_mlp_0_0_output in {26, 58}:
            return k_mlp_0_0_output == 18
        elif q_mlp_0_0_output in {27, 39}:
            return k_mlp_0_0_output == 15
        elif q_mlp_0_0_output in {36, 29}:
            return k_mlp_0_0_output == 50
        elif q_mlp_0_0_output in {30}:
            return k_mlp_0_0_output == 0
        elif q_mlp_0_0_output in {32}:
            return k_mlp_0_0_output == 31
        elif q_mlp_0_0_output in {34}:
            return k_mlp_0_0_output == 46
        elif q_mlp_0_0_output in {35}:
            return k_mlp_0_0_output == 6
        elif q_mlp_0_0_output in {40}:
            return k_mlp_0_0_output == 8
        elif q_mlp_0_0_output in {41, 54}:
            return k_mlp_0_0_output == 49
        elif q_mlp_0_0_output in {44}:
            return k_mlp_0_0_output == 27
        elif q_mlp_0_0_output in {45}:
            return k_mlp_0_0_output == 30
        elif q_mlp_0_0_output in {46}:
            return k_mlp_0_0_output == 53
        elif q_mlp_0_0_output in {48}:
            return k_mlp_0_0_output == 24
        elif q_mlp_0_0_output in {51}:
            return k_mlp_0_0_output == 16
        elif q_mlp_0_0_output in {55}:
            return k_mlp_0_0_output == 20
        elif q_mlp_0_0_output in {56, 57}:
            return k_mlp_0_0_output == 36

    num_attn_1_3_pattern = select(mlp_0_0_outputs, mlp_0_0_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, attn_0_2_output):
        key = (attn_1_2_output, attn_0_2_output)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "}"),
            (")", "("),
            (")", "<s>"),
            (")", "}"),
            ("}", "("),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 20
        elif key in {("{", "<s>")}:
            return 5
        elif key in {("<s>", "<s>")}:
            return 18
        return 3

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_3_output, attn_1_1_output):
        key = (attn_1_3_output, attn_1_1_output)
        if key in {
            (0, "<s>"),
            (0, "}"),
            (1, "<s>"),
            (2, "<s>"),
            (2, "}"),
            (3, "<s>"),
            (3, "}"),
            (4, "<s>"),
            (4, "}"),
            (5, "<s>"),
            (5, "}"),
            (6, "<s>"),
            (6, "}"),
            (7, "<s>"),
            (7, "}"),
            (8, "<s>"),
            (8, "}"),
            (9, ")"),
            (9, "<s>"),
            (9, "}"),
            (10, "<s>"),
            (10, "}"),
            (11, "<s>"),
            (11, "}"),
            (12, "<s>"),
            (12, "}"),
            (13, "<s>"),
            (13, "}"),
            (14, "<s>"),
            (14, "}"),
            (15, "<s>"),
            (15, "}"),
            (16, "<s>"),
            (16, "}"),
            (17, "<s>"),
            (17, "}"),
            (18, "<s>"),
            (18, "}"),
            (19, "<s>"),
            (19, "}"),
            (20, "<s>"),
            (20, "}"),
            (21, "<s>"),
            (21, "}"),
            (22, "<s>"),
            (22, "}"),
            (23, "<s>"),
            (23, "}"),
            (24, "<s>"),
            (24, "}"),
            (25, "<s>"),
            (25, "}"),
            (26, "<s>"),
            (26, "}"),
            (27, "<s>"),
            (27, "}"),
            (28, "<s>"),
            (28, "}"),
            (29, "<s>"),
            (29, "}"),
            (30, "<s>"),
            (30, "}"),
            (31, "<s>"),
            (31, "}"),
            (32, "<s>"),
            (32, "}"),
            (33, "<s>"),
            (33, "}"),
            (34, "<s>"),
            (34, "}"),
            (35, "<s>"),
            (35, "}"),
            (36, "<s>"),
            (36, "}"),
            (37, "<s>"),
            (37, "}"),
            (38, "<s>"),
            (38, "}"),
            (40, ")"),
            (40, "<s>"),
            (40, "}"),
            (41, "<s>"),
            (41, "}"),
            (42, "<s>"),
            (42, "}"),
            (43, "<s>"),
            (43, "}"),
            (44, "<s>"),
            (44, "}"),
            (45, "<s>"),
            (45, "}"),
            (46, "<s>"),
            (46, "}"),
            (47, ")"),
            (47, "<s>"),
            (47, "}"),
            (48, "<s>"),
            (48, "}"),
            (49, "<s>"),
            (49, "}"),
            (50, "<s>"),
            (50, "}"),
            (51, "<s>"),
            (51, "}"),
            (52, "<s>"),
            (52, "}"),
            (53, "<s>"),
            (53, "}"),
            (54, "<s>"),
            (54, "}"),
            (55, "<s>"),
            (55, "}"),
            (56, "<s>"),
            (56, "}"),
            (57, "<s>"),
            (57, "}"),
            (58, "<s>"),
            (58, "}"),
            (59, "("),
            (59, ")"),
            (59, "<s>"),
            (59, "}"),
        }:
            return 11
        elif key in {(1, "}"), (33, ")")}:
            return 36
        return 23

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_2_output, attn_1_1_output):
        key = (attn_1_2_output, attn_1_1_output)
        if key in {
            (")", ")"),
            (")", "<s>"),
            ("<s>", ")"),
            ("<s>", "<s>"),
            ("<s>", "}"),
            ("}", ")"),
            ("}", "<s>"),
        }:
            return 40
        elif key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 51
        elif key in {(")", "(")}:
            return 50
        return 3

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_1_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(token, position):
        key = (token, position)
        if key in {
            ("(", 0),
            ("(", 1),
            ("(", 4),
            ("(", 5),
            ("(", 6),
            ("(", 7),
            ("(", 9),
            ("(", 11),
            ("(", 12),
            ("(", 13),
            ("(", 15),
            ("(", 17),
            ("(", 19),
            ("(", 20),
            ("(", 21),
            ("(", 22),
            ("(", 23),
            ("(", 24),
            ("(", 25),
            ("(", 26),
            ("(", 27),
            ("(", 28),
            ("(", 30),
            ("(", 31),
            ("(", 32),
            ("(", 33),
            ("(", 34),
            ("(", 35),
            ("(", 36),
            ("(", 37),
            ("(", 38),
            ("(", 39),
            ("(", 40),
            ("(", 41),
            ("(", 42),
            ("(", 43),
            ("(", 44),
            ("(", 45),
            ("(", 46),
            ("(", 47),
            ("(", 48),
            ("(", 49),
            ("(", 50),
            ("(", 51),
            ("(", 52),
            ("(", 53),
            ("(", 54),
            ("(", 55),
            ("(", 56),
            ("(", 57),
            ("(", 58),
            ("(", 59),
            (")", 7),
            (")", 9),
            (")", 11),
            (")", 13),
            (")", 15),
            (")", 17),
            (")", 19),
            (")", 21),
            (")", 23),
            (")", 25),
            (")", 27),
            ("<s>", 7),
            ("<s>", 9),
            ("<s>", 11),
            ("<s>", 15),
            ("<s>", 17),
            ("<s>", 19),
            ("<s>", 21),
            ("<s>", 25),
            ("<s>", 27),
            ("{", 0),
            ("{", 1),
            ("{", 4),
            ("{", 5),
            ("{", 6),
            ("{", 7),
            ("{", 9),
            ("{", 11),
            ("{", 12),
            ("{", 13),
            ("{", 15),
            ("{", 17),
            ("{", 19),
            ("{", 20),
            ("{", 21),
            ("{", 22),
            ("{", 23),
            ("{", 24),
            ("{", 25),
            ("{", 26),
            ("{", 27),
            ("{", 28),
            ("{", 30),
            ("{", 31),
            ("{", 32),
            ("{", 33),
            ("{", 34),
            ("{", 35),
            ("{", 36),
            ("{", 37),
            ("{", 38),
            ("{", 39),
            ("{", 40),
            ("{", 41),
            ("{", 42),
            ("{", 43),
            ("{", 44),
            ("{", 45),
            ("{", 46),
            ("{", 47),
            ("{", 48),
            ("{", 49),
            ("{", 50),
            ("{", 51),
            ("{", 52),
            ("{", 53),
            ("{", 54),
            ("{", 55),
            ("{", 56),
            ("{", 57),
            ("{", 58),
            ("{", 59),
            ("}", 7),
            ("}", 9),
            ("}", 11),
            ("}", 13),
            ("}", 15),
            ("}", 17),
            ("}", 19),
            ("}", 21),
            ("}", 23),
            ("}", 25),
            ("}", 27),
        }:
            return 47
        elif key in {("<s>", 13), ("<s>", 23)}:
            return 40
        elif key in {
            ("(", 29),
            (")", 1),
            (")", 2),
            (")", 3),
            (")", 4),
            (")", 26),
            (")", 28),
            (")", 29),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 26),
            ("<s>", 28),
            ("<s>", 29),
            ("{", 29),
            ("}", 1),
            ("}", 2),
            ("}", 3),
            ("}", 4),
            ("}", 26),
            ("}", 28),
            ("}", 29),
        }:
            return 17
        elif key in {("(", 2), ("(", 3), ("{", 2), ("{", 3)}:
            return 36
        return 45

    mlp_1_3_outputs = [mlp_1_3(k0, k1) for k0, k1 in zip(tokens, positions)]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_1_0_output):
        key = (num_attn_1_2_output, num_attn_1_0_output)
        return 47

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output):
        key = num_attn_1_1_output
        return 16

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_2_output, num_attn_0_0_output):
        key = (num_attn_1_2_output, num_attn_0_0_output)
        return 23

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 7

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"{", "(", ")"}:
            return position == 2
        elif token in {"<s>", "}"}:
            return position == 7

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"{", "}", "("}:
            return position == 2
        elif token in {")"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 6

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, position):
        if token in {"("}:
            return position == 6
        elif token in {"}", ")"}:
            return position == 5
        elif token in {"<s>"}:
            return position == 7
        elif token in {"{"}:
            return position == 2

    attn_2_2_pattern = select_closest(positions, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, position):
        if token in {"("}:
            return position == 4
        elif token in {"{", "}", ")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 7

    attn_2_3_pattern = select_closest(positions, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_3_output, attn_1_1_output):
        if attn_1_3_output in {0, 32, 6, 12, 14, 25}:
            return attn_1_1_output == ""
        elif attn_1_3_output in {
            1,
            3,
            9,
            10,
            11,
            13,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            26,
            35,
            36,
            38,
            39,
            42,
            44,
            45,
            46,
            48,
            54,
            55,
            57,
            59,
        }:
            return attn_1_1_output == "}"
        elif attn_1_3_output in {
            2,
            4,
            5,
            7,
            8,
            15,
            16,
            24,
            27,
            28,
            29,
            30,
            31,
            33,
            34,
            37,
            40,
            41,
            43,
            47,
            49,
            50,
            51,
            52,
            53,
            56,
            58,
        }:
            return attn_1_1_output == ")"

    num_attn_2_0_pattern = select(attn_1_1_outputs, attn_1_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_3_output, attn_1_1_output):
        if attn_1_3_output in {
            0,
            1,
            2,
            4,
            6,
            7,
            10,
            12,
            17,
            21,
            22,
            23,
            26,
            29,
            31,
            33,
            34,
            37,
            41,
            44,
            48,
            49,
            50,
            51,
            54,
            57,
            58,
        }:
            return attn_1_1_output == "}"
        elif attn_1_3_output in {
            3,
            5,
            8,
            9,
            11,
            13,
            15,
            16,
            20,
            24,
            25,
            27,
            28,
            30,
            32,
            36,
            39,
            40,
            45,
            46,
            47,
            52,
            53,
            56,
            59,
        }:
            return attn_1_1_output == ""
        elif attn_1_3_output in {35, 38, 43, 14, 18, 19}:
            return attn_1_1_output == ")"
        elif attn_1_3_output in {42, 55}:
            return attn_1_1_output == "<s>"

    num_attn_2_1_pattern = select(attn_1_1_outputs, attn_1_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_3_output, attn_1_1_output):
        if attn_0_3_output in {"{", "("}:
            return attn_1_1_output == ""
        elif attn_0_3_output in {")"}:
            return attn_1_1_output == "("
        elif attn_0_3_output in {"<s>", "}"}:
            return attn_1_1_output == "{"

    num_attn_2_2_pattern = select(attn_1_1_outputs, attn_0_3_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_3_output, attn_1_0_output):
        if attn_1_3_output in {
            0,
            2,
            5,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            20,
            23,
            24,
            25,
            26,
            28,
            29,
            32,
            35,
            36,
            37,
            39,
            40,
            41,
            43,
            44,
            45,
            46,
            48,
            49,
            51,
            55,
        }:
            return attn_1_0_output == 4
        elif attn_1_3_output in {1, 15}:
            return attn_1_0_output == 30
        elif attn_1_3_output in {3}:
            return attn_1_0_output == 26
        elif attn_1_3_output in {4}:
            return attn_1_0_output == 56
        elif attn_1_3_output in {7}:
            return attn_1_0_output == 20
        elif attn_1_3_output in {19}:
            return attn_1_0_output == 44
        elif attn_1_3_output in {21}:
            return attn_1_0_output == 34
        elif attn_1_3_output in {38, 57, 22}:
            return attn_1_0_output == 6
        elif attn_1_3_output in {27}:
            return attn_1_0_output == 10
        elif attn_1_3_output in {30}:
            return attn_1_0_output == 48
        elif attn_1_3_output in {34, 54, 31}:
            return attn_1_0_output == 12
        elif attn_1_3_output in {33, 42, 52}:
            return attn_1_0_output == 14
        elif attn_1_3_output in {47}:
            return attn_1_0_output == 0
        elif attn_1_3_output in {50}:
            return attn_1_0_output == 1
        elif attn_1_3_output in {53}:
            return attn_1_0_output == 41
        elif attn_1_3_output in {56}:
            return attn_1_0_output == 29
        elif attn_1_3_output in {58}:
            return attn_1_0_output == 42
        elif attn_1_3_output in {59}:
            return attn_1_0_output == 52

    num_attn_2_3_pattern = select(attn_1_0_outputs, attn_1_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_3_output, attn_1_2_output):
        key = (attn_1_3_output, attn_1_2_output)
        if key in {
            (0, ")"),
            (2, ")"),
            (12, ")"),
            (14, ")"),
            (17, ")"),
            (20, ")"),
            (25, ")"),
            (26, ")"),
            (28, ")"),
            (30, ")"),
            (31, ")"),
            (31, "<s>"),
            (31, "{"),
            (31, "}"),
            (32, ")"),
            (34, ")"),
            (37, ")"),
            (37, "<s>"),
            (38, ")"),
            (40, ")"),
            (42, ")"),
            (43, ")"),
            (44, ")"),
            (48, ")"),
            (59, ")"),
        }:
            return 23
        return 37

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_2_output, mlp_0_2_output):
        key = (attn_2_2_output, mlp_0_2_output)
        if key in {
            (0, 17),
            (0, 20),
            (0, 46),
            (0, 55),
            (0, 57),
            (2, 17),
            (2, 20),
            (2, 46),
            (2, 53),
            (2, 55),
            (2, 57),
            (3, 17),
            (3, 46),
            (3, 55),
            (3, 57),
            (4, 17),
            (4, 55),
            (6, 17),
            (6, 20),
            (6, 46),
            (6, 55),
            (6, 57),
            (7, 20),
            (7, 46),
            (7, 55),
            (7, 57),
            (8, 17),
            (8, 20),
            (8, 46),
            (8, 55),
            (8, 57),
            (9, 17),
            (9, 55),
            (10, 17),
            (10, 20),
            (10, 46),
            (10, 55),
            (10, 57),
            (11, 17),
            (11, 20),
            (11, 46),
            (11, 55),
            (11, 57),
            (12, 17),
            (12, 20),
            (12, 46),
            (12, 53),
            (12, 55),
            (12, 57),
            (13, 17),
            (13, 20),
            (13, 46),
            (13, 55),
            (13, 57),
            (14, 17),
            (14, 20),
            (14, 37),
            (14, 42),
            (14, 46),
            (14, 47),
            (14, 53),
            (14, 54),
            (14, 55),
            (14, 57),
            (15, 17),
            (15, 20),
            (15, 46),
            (15, 55),
            (15, 57),
            (16, 17),
            (16, 46),
            (16, 55),
            (16, 57),
            (17, 17),
            (17, 46),
            (17, 55),
            (17, 57),
            (18, 17),
            (18, 46),
            (18, 55),
            (18, 57),
            (19, 17),
            (19, 20),
            (19, 46),
            (19, 47),
            (19, 53),
            (19, 54),
            (19, 55),
            (19, 57),
            (20, 3),
            (20, 17),
            (20, 20),
            (20, 32),
            (20, 34),
            (20, 37),
            (20, 42),
            (20, 46),
            (20, 47),
            (20, 48),
            (20, 53),
            (20, 54),
            (20, 55),
            (20, 57),
            (21, 17),
            (21, 55),
            (22, 17),
            (22, 46),
            (22, 55),
            (22, 57),
            (23, 17),
            (23, 20),
            (23, 37),
            (23, 46),
            (23, 47),
            (23, 53),
            (23, 54),
            (23, 55),
            (23, 57),
            (24, 17),
            (24, 46),
            (24, 55),
            (24, 57),
            (25, 17),
            (25, 46),
            (25, 55),
            (25, 57),
            (26, 17),
            (26, 20),
            (26, 37),
            (26, 46),
            (26, 47),
            (26, 53),
            (26, 54),
            (26, 55),
            (26, 57),
            (27, 17),
            (27, 46),
            (27, 55),
            (27, 57),
            (28, 17),
            (28, 46),
            (28, 55),
            (28, 57),
            (29, 17),
            (29, 46),
            (29, 55),
            (29, 57),
            (30, 17),
            (30, 20),
            (30, 46),
            (30, 47),
            (30, 53),
            (30, 54),
            (30, 55),
            (30, 57),
            (31, 17),
            (32, 46),
            (32, 55),
            (32, 57),
            (33, 17),
            (33, 46),
            (33, 55),
            (33, 57),
            (34, 17),
            (34, 46),
            (34, 55),
            (34, 57),
            (35, 17),
            (35, 46),
            (35, 55),
            (35, 57),
            (36, 17),
            (36, 46),
            (36, 55),
            (36, 57),
            (37, 17),
            (37, 20),
            (37, 46),
            (37, 55),
            (37, 57),
            (38, 17),
            (38, 20),
            (38, 46),
            (38, 55),
            (38, 57),
            (40, 17),
            (41, 17),
            (41, 46),
            (41, 55),
            (41, 57),
            (42, 17),
            (42, 46),
            (42, 55),
            (42, 57),
            (43, 17),
            (44, 17),
            (44, 46),
            (44, 55),
            (44, 57),
            (45, 17),
            (45, 46),
            (45, 55),
            (45, 57),
            (46, 17),
            (46, 46),
            (46, 55),
            (46, 57),
            (47, 17),
            (47, 46),
            (47, 55),
            (47, 57),
            (48, 17),
            (48, 20),
            (48, 46),
            (48, 55),
            (48, 57),
            (49, 17),
            (49, 20),
            (49, 46),
            (49, 55),
            (49, 57),
            (50, 17),
            (50, 46),
            (50, 55),
            (50, 57),
            (51, 17),
            (51, 20),
            (51, 46),
            (51, 47),
            (51, 53),
            (51, 54),
            (51, 55),
            (51, 57),
            (52, 17),
            (52, 46),
            (52, 55),
            (52, 57),
            (53, 17),
            (53, 46),
            (53, 55),
            (53, 57),
            (54, 17),
            (54, 46),
            (54, 55),
            (54, 57),
            (55, 17),
            (55, 46),
            (55, 55),
            (55, 57),
            (56, 3),
            (56, 17),
            (56, 20),
            (56, 32),
            (56, 34),
            (56, 37),
            (56, 42),
            (56, 46),
            (56, 47),
            (56, 48),
            (56, 53),
            (56, 54),
            (56, 55),
            (56, 57),
            (57, 17),
            (57, 20),
            (57, 37),
            (57, 46),
            (57, 47),
            (57, 53),
            (57, 54),
            (57, 55),
            (57, 57),
            (58, 17),
            (58, 46),
            (58, 55),
            (58, 57),
        }:
            return 29
        elif key in {
            (0, 35),
            (0, 47),
            (0, 48),
            (0, 52),
            (0, 53),
            (0, 54),
            (1, 8),
            (1, 11),
            (1, 17),
            (1, 55),
            (1, 59),
            (2, 37),
            (2, 47),
            (2, 54),
            (3, 35),
            (3, 48),
            (3, 52),
            (4, 46),
            (4, 57),
            (5, 11),
            (5, 17),
            (5, 35),
            (5, 55),
            (5, 59),
            (6, 11),
            (6, 53),
            (6, 59),
            (7, 7),
            (7, 8),
            (7, 21),
            (7, 27),
            (7, 33),
            (7, 45),
            (7, 53),
            (8, 35),
            (8, 48),
            (8, 52),
            (9, 9),
            (9, 48),
            (9, 52),
            (10, 53),
            (11, 35),
            (11, 48),
            (12, 37),
            (12, 47),
            (12, 54),
            (13, 9),
            (13, 48),
            (13, 52),
            (13, 53),
            (14, 3),
            (14, 34),
            (14, 35),
            (14, 48),
            (14, 59),
            (15, 35),
            (15, 47),
            (15, 48),
            (15, 52),
            (15, 53),
            (15, 54),
            (16, 20),
            (16, 35),
            (16, 48),
            (16, 52),
            (17, 20),
            (17, 48),
            (17, 52),
            (18, 20),
            (18, 35),
            (18, 48),
            (18, 52),
            (19, 37),
            (20, 1),
            (20, 11),
            (20, 15),
            (20, 22),
            (21, 9),
            (21, 13),
            (21, 46),
            (21, 57),
            (22, 20),
            (22, 48),
            (22, 52),
            (23, 8),
            (23, 11),
            (23, 42),
            (23, 59),
            (24, 11),
            (24, 20),
            (24, 35),
            (24, 59),
            (25, 35),
            (25, 59),
            (26, 35),
            (26, 42),
            (26, 48),
            (27, 9),
            (27, 20),
            (27, 52),
            (28, 20),
            (28, 52),
            (29, 9),
            (29, 13),
            (29, 20),
            (29, 48),
            (29, 52),
            (30, 37),
            (32, 8),
            (32, 11),
            (32, 17),
            (33, 20),
            (33, 35),
            (33, 59),
            (34, 9),
            (34, 13),
            (34, 20),
            (34, 25),
            (34, 35),
            (34, 36),
            (34, 41),
            (34, 53),
            (34, 59),
            (35, 20),
            (35, 35),
            (35, 48),
            (35, 52),
            (36, 20),
            (37, 9),
            (37, 35),
            (37, 47),
            (37, 48),
            (37, 52),
            (37, 53),
            (38, 48),
            (38, 52),
            (39, 11),
            (39, 59),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 8),
            (40, 9),
            (40, 10),
            (40, 11),
            (40, 12),
            (40, 13),
            (40, 14),
            (40, 15),
            (40, 16),
            (40, 18),
            (40, 19),
            (40, 20),
            (40, 21),
            (40, 22),
            (40, 23),
            (40, 24),
            (40, 25),
            (40, 26),
            (40, 27),
            (40, 28),
            (40, 29),
            (40, 30),
            (40, 31),
            (40, 32),
            (40, 33),
            (40, 34),
            (40, 35),
            (40, 36),
            (40, 37),
            (40, 38),
            (40, 39),
            (40, 40),
            (40, 41),
            (40, 42),
            (40, 43),
            (40, 44),
            (40, 45),
            (40, 46),
            (40, 47),
            (40, 48),
            (40, 49),
            (40, 50),
            (40, 51),
            (40, 52),
            (40, 53),
            (40, 54),
            (40, 55),
            (40, 57),
            (40, 58),
            (40, 59),
            (42, 9),
            (42, 20),
            (42, 48),
            (42, 52),
            (43, 52),
            (43, 55),
            (44, 20),
            (45, 9),
            (45, 11),
            (45, 13),
            (45, 20),
            (45, 35),
            (45, 59),
            (46, 20),
            (46, 52),
            (47, 9),
            (47, 20),
            (47, 35),
            (47, 48),
            (47, 52),
            (48, 9),
            (48, 48),
            (48, 52),
            (48, 53),
            (49, 47),
            (49, 53),
            (50, 9),
            (50, 20),
            (50, 52),
            (51, 37),
            (51, 52),
            (52, 20),
            (53, 20),
            (53, 35),
            (53, 48),
            (53, 59),
            (54, 20),
            (55, 9),
            (55, 20),
            (55, 35),
            (55, 48),
            (55, 52),
            (56, 1),
            (56, 11),
            (56, 15),
            (56, 22),
            (57, 3),
            (57, 9),
            (57, 13),
            (57, 42),
            (57, 48),
            (57, 52),
            (58, 52),
            (59, 52),
        }:
            return 58
        elif key in {
            (1, 35),
            (1, 48),
            (1, 52),
            (5, 48),
            (5, 52),
            (6, 35),
            (6, 48),
            (6, 52),
            (7, 11),
            (7, 35),
            (7, 48),
            (7, 52),
            (7, 59),
            (11, 52),
            (14, 52),
            (23, 35),
            (23, 48),
            (23, 52),
            (24, 48),
            (24, 52),
            (25, 48),
            (25, 52),
            (26, 52),
            (32, 35),
            (32, 48),
            (32, 52),
            (32, 59),
            (33, 48),
            (33, 52),
            (34, 48),
            (34, 52),
            (39, 17),
            (39, 35),
            (39, 48),
            (39, 52),
            (45, 48),
            (45, 52),
            (53, 52),
        }:
            return 44
        elif key in {(7, 17)}:
            return 38
        return 50

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_2_outputs, mlp_0_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_1_output, attn_2_0_output):
        key = (attn_2_1_output, attn_2_0_output)
        if key in {
            (")", ")"),
            (")", "<s>"),
            (")", "}"),
            ("<s>", ")"),
            ("{", ")"),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 21
        return 19

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_0_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_0_0_output, attn_0_1_output):
        key = (attn_0_0_output, attn_0_1_output)
        if key in {
            ("(", "{"),
            (")", "{"),
            ("<s>", "{"),
            ("{", "("),
            ("{", ")"),
            ("{", "<s>"),
            ("{", "{"),
            ("{", "}"),
            ("}", "{"),
        }:
            return 44
        return 37

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_1_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output, num_attn_2_1_output):
        key = (num_attn_1_0_output, num_attn_2_1_output)
        return 20

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_2_output, num_attn_2_2_output):
        key = (num_attn_1_2_output, num_attn_2_2_output)
        return 4

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_0_output, num_attn_1_2_output):
        key = (num_attn_2_0_output, num_attn_1_2_output)
        return 54

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_1_output, num_attn_1_2_output):
        key = (num_attn_2_1_output, num_attn_1_2_output)
        return 58

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_3_output_scores = classifier_weights.loc[
        [("num_mlp_2_3_outputs", str(v)) for v in num_mlp_2_3_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                mlp_0_2_output_scores,
                mlp_0_3_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                num_mlp_0_2_output_scores,
                num_mlp_0_3_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                mlp_1_2_output_scores,
                mlp_1_3_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                num_mlp_1_2_output_scores,
                num_mlp_1_3_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                mlp_2_2_output_scores,
                mlp_2_3_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                num_mlp_2_2_output_scores,
                num_mlp_2_3_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(
    run(
        [
            "<s>",
            "(",
            "(",
            "{",
            "{",
            "{",
            "(",
            "{",
            "}",
            ")",
            "}",
            "}",
            "(",
            ")",
            "{",
            "}",
            "}",
            "(",
            ")",
            "{",
            "}",
            "(",
            ")",
            "{",
            "}",
            ")",
            "(",
            ")",
            ")",
            "(",
        ]
    )
)
