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
        "output/length/rasp/dyck1/trainlength40/s2/dyck1_weights.csv",
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
        if q_position in {0}:
            return k_position == 79
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {8, 2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 68
        elif q_position in {39, 6, 31}:
            return k_position == 5
        elif q_position in {20, 7}:
            return k_position == 19
        elif q_position in {9, 10}:
            return k_position == 8
        elif q_position in {16, 32, 11, 34}:
            return k_position == 10
        elif q_position in {12, 23, 26, 28, 30}:
            return k_position == 9
        elif q_position in {27, 13, 15}:
            return k_position == 12
        elif q_position in {49, 77, 14, 70}:
            return k_position == 13
        elif q_position in {56, 17, 58, 65}:
            return k_position == 16
        elif q_position in {18, 71}:
            return k_position == 17
        elif q_position in {42, 19}:
            return k_position == 14
        elif q_position in {64, 44, 21, 62}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {24, 63}:
            return k_position == 23
        elif q_position in {25, 67}:
            return k_position == 22
        elif q_position in {66, 29}:
            return k_position == 28
        elif q_position in {33, 75}:
            return k_position == 32
        elif q_position in {40, 59, 35, 55}:
            return k_position == 34
        elif q_position in {36}:
            return k_position == 29
        elif q_position in {37}:
            return k_position == 7
        elif q_position in {69, 38, 79}:
            return k_position == 36
        elif q_position in {41, 43}:
            return k_position == 24
        elif q_position in {45}:
            return k_position == 52
        elif q_position in {46}:
            return k_position == 70
        elif q_position in {68, 47}:
            return k_position == 75
        elif q_position in {48}:
            return k_position == 27
        elif q_position in {50}:
            return k_position == 39
        elif q_position in {51}:
            return k_position == 64
        elif q_position in {74, 52, 78}:
            return k_position == 49
        elif q_position in {53}:
            return k_position == 65
        elif q_position in {54}:
            return k_position == 42
        elif q_position in {57}:
            return k_position == 69
        elif q_position in {60}:
            return k_position == 62
        elif q_position in {61}:
            return k_position == 72
        elif q_position in {72}:
            return k_position == 41
        elif q_position in {73}:
            return k_position == 43
        elif q_position in {76}:
            return k_position == 31

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 51

    attn_0_1_pattern = select_closest(positions, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 68

    attn_0_2_pattern = select_closest(positions, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 40
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 29}:
            return k_position == 10
        elif q_position in {3, 4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 27
        elif q_position in {33, 6, 39}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 77
        elif q_position in {37, 8, 10, 23, 26}:
            return k_position == 7
        elif q_position in {9, 22, 17, 15}:
            return k_position == 8
        elif q_position in {48, 11}:
            return k_position == 15
        elif q_position in {32, 34, 36, 69, 71, 42, 75, 12, 59}:
            return k_position == 11
        elif q_position in {40, 51, 19, 13}:
            return k_position == 18
        elif q_position in {14, 20, 21, 28, 30}:
            return k_position == 9
        elif q_position in {35, 16, 24, 25, 61}:
            return k_position == 5
        elif q_position in {18, 66}:
            return k_position == 17
        elif q_position in {57, 27}:
            return k_position == 21
        elif q_position in {60, 31}:
            return k_position == 25
        elif q_position in {38}:
            return k_position == 72
        elif q_position in {41}:
            return k_position == 19
        elif q_position in {64, 65, 43, 79}:
            return k_position == 68
        elif q_position in {44, 45}:
            return k_position == 73
        elif q_position in {74, 46}:
            return k_position == 70
        elif q_position in {56, 73, 47}:
            return k_position == 23
        elif q_position in {49, 54}:
            return k_position == 62
        elif q_position in {50, 58}:
            return k_position == 13
        elif q_position in {52, 76}:
            return k_position == 78
        elif q_position in {53, 78, 63}:
            return k_position == 58
        elif q_position in {55}:
            return k_position == 47
        elif q_position in {62}:
            return k_position == 49
        elif q_position in {67, 70}:
            return k_position == 16
        elif q_position in {68}:
            return k_position == 43
        elif q_position in {72}:
            return k_position == 20
        elif q_position in {77}:
            return k_position == 45

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 2

    attn_0_4_pattern = select_closest(positions, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 2, 36}:
            return k_position == 35
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {33, 4}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 4
        elif q_position in {42, 51, 77, 7}:
            return k_position == 49
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10, 11, 74, 52, 58, 60, 63}:
            return k_position == 9
        elif q_position in {59, 12}:
            return k_position == 11
        elif q_position in {56, 13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16, 65}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {73, 20, 70}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {79, 23}:
            return k_position == 22
        elif q_position in {24, 44}:
            return k_position == 23
        elif q_position in {25, 39}:
            return k_position == 5
        elif q_position in {26, 71}:
            return k_position == 25
        elif q_position in {64, 27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {62, 30}:
            return k_position == 29
        elif q_position in {31}:
            return k_position == 30
        elif q_position in {32, 69}:
            return k_position == 31
        elif q_position in {34}:
            return k_position == 33
        elif q_position in {35}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 36
        elif q_position in {40, 78, 38}:
            return k_position == 37
        elif q_position in {41}:
            return k_position == 43
        elif q_position in {43, 54}:
            return k_position == 38
        elif q_position in {50, 45}:
            return k_position == 70
        elif q_position in {46}:
            return k_position == 67
        elif q_position in {47}:
            return k_position == 41
        elif q_position in {48, 66}:
            return k_position == 32
        elif q_position in {49}:
            return k_position == 79
        elif q_position in {53}:
            return k_position == 56
        elif q_position in {55}:
            return k_position == 78
        elif q_position in {57}:
            return k_position == 42
        elif q_position in {61}:
            return k_position == 77
        elif q_position in {72, 67}:
            return k_position == 44
        elif q_position in {68}:
            return k_position == 51
        elif q_position in {75}:
            return k_position == 45
        elif q_position in {76}:
            return k_position == 61

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 9, 43, 45, 78, 54, 61, 62}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {33, 3, 37}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 18
        elif q_position in {6, 8, 76, 47, 51}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 54
        elif q_position in {10, 11, 12, 39}:
            return k_position == 8
        elif q_position in {32, 13, 14}:
            return k_position == 11
        elif q_position in {17, 60, 15}:
            return k_position == 13
        elif q_position in {16, 19}:
            return k_position == 14
        elif q_position in {64, 34, 49, 18, 21, 55, 23}:
            return k_position == 9
        elif q_position in {20, 71}:
            return k_position == 19
        elif q_position in {22}:
            return k_position == 20
        elif q_position in {24}:
            return k_position == 21
        elif q_position in {25}:
            return k_position == 22
        elif q_position in {26, 68}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28, 31}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {38, 30}:
            return k_position == 24
        elif q_position in {35}:
            return k_position == 34
        elif q_position in {36}:
            return k_position == 10
        elif q_position in {40, 69}:
            return k_position == 76
        elif q_position in {41}:
            return k_position == 41
        elif q_position in {42}:
            return k_position == 72
        elif q_position in {44, 53}:
            return k_position == 64
        elif q_position in {70, 46}:
            return k_position == 17
        elif q_position in {48}:
            return k_position == 75
        elif q_position in {50}:
            return k_position == 23
        elif q_position in {75, 52}:
            return k_position == 66
        elif q_position in {56, 67}:
            return k_position == 31
        elif q_position in {57}:
            return k_position == 78
        elif q_position in {58}:
            return k_position == 61
        elif q_position in {59}:
            return k_position == 47
        elif q_position in {77, 63}:
            return k_position == 44
        elif q_position in {65}:
            return k_position == 50
        elif q_position in {66}:
            return k_position == 33
        elif q_position in {72}:
            return k_position == 57
        elif q_position in {73}:
            return k_position == 65
        elif q_position in {74, 79}:
            return k_position == 16

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(token, position):
        if token in {"("}:
            return position == 9
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 3

    attn_0_7_pattern = select_closest(positions, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"("}:
            return position == 34
        elif token in {")"}:
            return position == 72
        elif token in {"<s>"}:
            return position == 1

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"("}:
            return position == 4
        elif token in {")"}:
            return position == 65
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 38
        elif token in {")"}:
            return position == 0
        elif token in {"<s>"}:
            return position == 3

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"("}:
            return position == 10
        elif token in {")"}:
            return position == 62
        elif token in {"<s>"}:
            return position == 31

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"("}:
            return position == 65
        elif token in {")"}:
            return position == 39
        elif token in {"<s>"}:
            return position == 79

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"("}:
            return position == 43
        elif token in {")"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 9

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {
            0,
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
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return token == ""
        elif position in {1, 2, 15, 19, 21}:
            return token == "<s>"
        elif position in {3, 4, 5, 7}:
            return token == ")"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"("}:
            return position == 12
        elif token in {")"}:
            return position == 59
        elif token in {"<s>"}:
            return position == 15

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_2_output):
        key = (attn_0_0_output, attn_0_2_output)
        if key in {("(", ")"), (")", "("), (")", ")"), (")", "<s>")}:
            return 5
        elif key in {("<s>", ")")}:
            return 7
        elif key in {("<s>", "<s>")}:
            return 32
        return 14

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_5_output, attn_0_3_output):
        key = (attn_0_5_output, attn_0_3_output)
        if key in {
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            ("<s>", "("),
            ("<s>", ")"),
            ("<s>", "<s>"),
        }:
            return 2
        return 13

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_3_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_4_output, num_attn_0_2_output):
        key = (num_attn_0_4_output, num_attn_0_2_output)
        return 3

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_7_output, num_attn_0_6_output):
        key = (num_attn_0_7_output, num_attn_0_6_output)
        return 16

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_0_output, position):
        if attn_0_0_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_0_output in {")"}:
            return position == 5

    attn_1_0_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_2_output, mlp_0_1_output):
        if attn_0_2_output in {"(", "<s>"}:
            return mlp_0_1_output == 1
        elif attn_0_2_output in {")"}:
            return mlp_0_1_output == 45

    attn_1_1_pattern = select_closest(mlp_0_1_outputs, attn_0_2_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_7_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_0_output, position):
        if attn_0_0_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_0_output in {")"}:
            return position == 13

    attn_1_2_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_6_output, position):
        if attn_0_6_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_6_output in {")"}:
            return position == 11

    attn_1_3_pattern = select_closest(positions, attn_0_6_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 12
        elif token in {"<s>"}:
            return position == 5

    attn_1_4_pattern = select_closest(positions, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_1_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_6_output, position):
        if attn_0_6_output in {"("}:
            return position == 1
        elif attn_0_6_output in {")"}:
            return position == 8
        elif attn_0_6_output in {"<s>"}:
            return position == 3

    attn_1_5_pattern = select_closest(positions, attn_0_6_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_7_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_2_output, position):
        if attn_0_2_output in {"(", "<s>"}:
            return position == 3
        elif attn_0_2_output in {")"}:
            return position == 9

    attn_1_6_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_4_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_7_output, position):
        if attn_0_7_output in {"("}:
            return position == 3
        elif attn_0_7_output in {")"}:
            return position == 9
        elif attn_0_7_output in {"<s>"}:
            return position == 1

    attn_1_7_pattern = select_closest(positions, attn_0_7_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_0_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_1_output, num_mlp_0_0_output):
        if mlp_0_1_output in {0}:
            return num_mlp_0_0_output == 1
        elif mlp_0_1_output in {1}:
            return num_mlp_0_0_output == 54
        elif mlp_0_1_output in {32, 2, 72, 73, 13, 15, 48, 60}:
            return num_mlp_0_0_output == 71
        elif mlp_0_1_output in {3, 12, 45, 14, 47, 17, 50, 57, 59, 61}:
            return num_mlp_0_0_output == 64
        elif mlp_0_1_output in {4}:
            return num_mlp_0_0_output == 20
        elif mlp_0_1_output in {56, 5, 22}:
            return num_mlp_0_0_output == 31
        elif mlp_0_1_output in {6, 41, 75, 77, 16, 20, 30}:
            return num_mlp_0_0_output == 14
        elif mlp_0_1_output in {36, 38, 7, 42, 74, 21, 55, 23, 26, 31}:
            return num_mlp_0_0_output == 46
        elif mlp_0_1_output in {8}:
            return num_mlp_0_0_output == 25
        elif mlp_0_1_output in {66, 9, 44, 49, 53}:
            return num_mlp_0_0_output == 26
        elif mlp_0_1_output in {10}:
            return num_mlp_0_0_output == 69
        elif mlp_0_1_output in {11}:
            return num_mlp_0_0_output == 28
        elif mlp_0_1_output in {18, 62}:
            return num_mlp_0_0_output == 39
        elif mlp_0_1_output in {19}:
            return num_mlp_0_0_output == 15
        elif mlp_0_1_output in {24, 34}:
            return num_mlp_0_0_output == 67
        elif mlp_0_1_output in {25, 28, 70}:
            return num_mlp_0_0_output == 79
        elif mlp_0_1_output in {27, 68}:
            return num_mlp_0_0_output == 12
        elif mlp_0_1_output in {29, 78, 79}:
            return num_mlp_0_0_output == 33
        elif mlp_0_1_output in {33}:
            return num_mlp_0_0_output == 42
        elif mlp_0_1_output in {51, 35}:
            return num_mlp_0_0_output == 68
        elif mlp_0_1_output in {65, 37}:
            return num_mlp_0_0_output == 77
        elif mlp_0_1_output in {39}:
            return num_mlp_0_0_output == 70
        elif mlp_0_1_output in {40, 67}:
            return num_mlp_0_0_output == 44
        elif mlp_0_1_output in {58, 43, 63}:
            return num_mlp_0_0_output == 53
        elif mlp_0_1_output in {46}:
            return num_mlp_0_0_output == 60
        elif mlp_0_1_output in {52}:
            return num_mlp_0_0_output == 30
        elif mlp_0_1_output in {54}:
            return num_mlp_0_0_output == 24
        elif mlp_0_1_output in {64}:
            return num_mlp_0_0_output == 52
        elif mlp_0_1_output in {69, 71}:
            return num_mlp_0_0_output == 75
        elif mlp_0_1_output in {76}:
            return num_mlp_0_0_output == 34

    num_attn_1_0_pattern = select(
        num_mlp_0_0_outputs, mlp_0_1_outputs, num_predicate_1_0
    )
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_7_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_1_output, mlp_0_1_output):
        if num_mlp_0_1_output in {0, 77, 46}:
            return mlp_0_1_output == 65
        elif num_mlp_0_1_output in {1, 2, 37, 71, 42, 51, 52, 63}:
            return mlp_0_1_output == 0
        elif num_mlp_0_1_output in {
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            32,
            33,
            34,
            36,
            38,
            40,
            41,
            43,
            44,
            45,
            47,
            50,
            55,
            56,
            57,
            58,
            59,
            61,
            62,
            64,
            65,
            67,
            68,
            69,
            70,
            73,
            74,
            75,
            76,
            78,
            79,
        }:
            return mlp_0_1_output == 2
        elif num_mlp_0_1_output in {72, 60, 31}:
            return mlp_0_1_output == 60
        elif num_mlp_0_1_output in {35}:
            return mlp_0_1_output == 11
        elif num_mlp_0_1_output in {66, 39, 48, 49, 54}:
            return mlp_0_1_output == 42
        elif num_mlp_0_1_output in {53}:
            return mlp_0_1_output == 53

    num_attn_1_1_pattern = select(
        mlp_0_1_outputs, num_mlp_0_1_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_7_output, mlp_0_0_output):
        if attn_0_7_output in {"(", "<s>", ")"}:
            return mlp_0_0_output == 32

    num_attn_1_2_pattern = select(mlp_0_0_outputs, attn_0_7_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_6_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_0_output, mlp_0_0_output):
        if num_mlp_0_0_output in {0}:
            return mlp_0_0_output == 76
        elif num_mlp_0_0_output in {1, 36}:
            return mlp_0_0_output == 46
        elif num_mlp_0_0_output in {41, 2}:
            return mlp_0_0_output == 69
        elif num_mlp_0_0_output in {3, 68, 39, 7, 9, 42, 48, 25, 61, 62}:
            return mlp_0_0_output == 71
        elif num_mlp_0_0_output in {34, 4}:
            return mlp_0_0_output == 25
        elif num_mlp_0_0_output in {18, 20, 5, 70}:
            return mlp_0_0_output == 39
        elif num_mlp_0_0_output in {
            6,
            8,
            11,
            12,
            13,
            15,
            16,
            23,
            28,
            30,
            33,
            35,
            37,
            40,
            43,
            44,
            45,
            46,
            47,
            49,
            50,
            53,
            54,
            56,
            57,
            59,
            64,
            65,
            69,
            71,
            72,
            73,
            74,
            76,
            77,
            78,
            79,
        }:
            return mlp_0_0_output == 53
        elif num_mlp_0_0_output in {10}:
            return mlp_0_0_output == 42
        elif num_mlp_0_0_output in {66, 31, 14, 22}:
            return mlp_0_0_output == 14
        elif num_mlp_0_0_output in {17, 58, 51, 52}:
            return mlp_0_0_output == 64
        elif num_mlp_0_0_output in {19}:
            return mlp_0_0_output == 34
        elif num_mlp_0_0_output in {60, 21, 55}:
            return mlp_0_0_output == 68
        elif num_mlp_0_0_output in {24}:
            return mlp_0_0_output == 74
        elif num_mlp_0_0_output in {26}:
            return mlp_0_0_output == 33
        elif num_mlp_0_0_output in {27}:
            return mlp_0_0_output == 79
        elif num_mlp_0_0_output in {29}:
            return mlp_0_0_output == 16
        elif num_mlp_0_0_output in {32}:
            return mlp_0_0_output == 43
        elif num_mlp_0_0_output in {38}:
            return mlp_0_0_output == 70
        elif num_mlp_0_0_output in {63}:
            return mlp_0_0_output == 65
        elif num_mlp_0_0_output in {67}:
            return mlp_0_0_output == 51
        elif num_mlp_0_0_output in {75}:
            return mlp_0_0_output == 19

    num_attn_1_3_pattern = select(
        mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_7_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(num_mlp_0_1_output, mlp_0_0_output):
        if num_mlp_0_1_output in {
            0,
            1,
            3,
            5,
            8,
            11,
            14,
            15,
            16,
            17,
            20,
            21,
            25,
            27,
            29,
            30,
            31,
            34,
            40,
            43,
            45,
            47,
            51,
            54,
            55,
            57,
            60,
            61,
            63,
            67,
            68,
            69,
            71,
            73,
            75,
            77,
        }:
            return mlp_0_0_output == 32
        elif num_mlp_0_1_output in {32, 2, 35, 66, 70, 7, 41, 74, 46, 50, 19, 23, 62}:
            return mlp_0_0_output == 63
        elif num_mlp_0_1_output in {
            4,
            6,
            9,
            10,
            12,
            13,
            18,
            22,
            24,
            26,
            28,
            33,
            36,
            37,
            38,
            44,
            48,
            49,
            56,
            58,
            59,
            64,
            65,
            72,
            76,
            78,
        }:
            return mlp_0_0_output == 5
        elif num_mlp_0_1_output in {39}:
            return mlp_0_0_output == 42
        elif num_mlp_0_1_output in {42, 52, 79}:
            return mlp_0_0_output == 2
        elif num_mlp_0_1_output in {53}:
            return mlp_0_0_output == 57

    num_attn_1_4_pattern = select(
        mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_4
    )
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_5_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(num_mlp_0_1_output, mlp_0_0_output):
        if num_mlp_0_1_output in {0, 34, 5, 41, 47, 52}:
            return mlp_0_0_output == 63
        elif num_mlp_0_1_output in {1, 36, 57}:
            return mlp_0_0_output == 59
        elif num_mlp_0_1_output in {32, 2, 73, 13, 14, 46, 16, 21, 58, 60}:
            return mlp_0_0_output == 5
        elif num_mlp_0_1_output in {3}:
            return mlp_0_0_output == 61
        elif num_mlp_0_1_output in {4}:
            return mlp_0_0_output == 45
        elif num_mlp_0_1_output in {27, 6}:
            return mlp_0_0_output == 29
        elif num_mlp_0_1_output in {43, 7}:
            return mlp_0_0_output == 18
        elif num_mlp_0_1_output in {8, 76}:
            return mlp_0_0_output == 2
        elif num_mlp_0_1_output in {9}:
            return mlp_0_0_output == 65
        elif num_mlp_0_1_output in {10}:
            return mlp_0_0_output == 44
        elif num_mlp_0_1_output in {11}:
            return mlp_0_0_output == 70
        elif num_mlp_0_1_output in {35, 12, 79, 63}:
            return mlp_0_0_output == 15
        elif num_mlp_0_1_output in {44, 15}:
            return mlp_0_0_output == 20
        elif num_mlp_0_1_output in {17, 66}:
            return mlp_0_0_output == 67
        elif num_mlp_0_1_output in {18, 39}:
            return mlp_0_0_output == 74
        elif num_mlp_0_1_output in {59, 19, 69}:
            return mlp_0_0_output == 43
        elif num_mlp_0_1_output in {20}:
            return mlp_0_0_output == 26
        elif num_mlp_0_1_output in {25, 77, 22}:
            return mlp_0_0_output == 6
        elif num_mlp_0_1_output in {23}:
            return mlp_0_0_output == 54
        elif num_mlp_0_1_output in {24, 33, 64}:
            return mlp_0_0_output == 17
        elif num_mlp_0_1_output in {26, 78, 71}:
            return mlp_0_0_output == 3
        elif num_mlp_0_1_output in {28}:
            return mlp_0_0_output == 11
        elif num_mlp_0_1_output in {29}:
            return mlp_0_0_output == 53
        elif num_mlp_0_1_output in {53, 30}:
            return mlp_0_0_output == 4
        elif num_mlp_0_1_output in {37, 42, 50, 55, 31}:
            return mlp_0_0_output == 13
        elif num_mlp_0_1_output in {38}:
            return mlp_0_0_output == 0
        elif num_mlp_0_1_output in {40, 67}:
            return mlp_0_0_output == 22
        elif num_mlp_0_1_output in {45}:
            return mlp_0_0_output == 25
        elif num_mlp_0_1_output in {48}:
            return mlp_0_0_output == 32
        elif num_mlp_0_1_output in {49}:
            return mlp_0_0_output == 8
        elif num_mlp_0_1_output in {51}:
            return mlp_0_0_output == 40
        elif num_mlp_0_1_output in {54}:
            return mlp_0_0_output == 62
        elif num_mlp_0_1_output in {56}:
            return mlp_0_0_output == 9
        elif num_mlp_0_1_output in {68, 61}:
            return mlp_0_0_output == 51
        elif num_mlp_0_1_output in {62}:
            return mlp_0_0_output == 7
        elif num_mlp_0_1_output in {65}:
            return mlp_0_0_output == 41
        elif num_mlp_0_1_output in {75, 70}:
            return mlp_0_0_output == 50
        elif num_mlp_0_1_output in {72}:
            return mlp_0_0_output == 24
        elif num_mlp_0_1_output in {74}:
            return mlp_0_0_output == 23

    num_attn_1_5_pattern = select(
        mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_5
    )
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_6_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0, 2, 44, 50, 55, 56}:
            return k_mlp_0_1_output == 13
        elif q_mlp_0_1_output in {1}:
            return k_mlp_0_1_output == 22
        elif q_mlp_0_1_output in {64, 3, 8, 45, 54, 59}:
            return k_mlp_0_1_output == 79
        elif q_mlp_0_1_output in {25, 10, 4}:
            return k_mlp_0_1_output == 24
        elif q_mlp_0_1_output in {40, 75, 5}:
            return k_mlp_0_1_output == 64
        elif q_mlp_0_1_output in {6, 7}:
            return k_mlp_0_1_output == 18
        elif q_mlp_0_1_output in {9, 63, 47}:
            return k_mlp_0_1_output == 39
        elif q_mlp_0_1_output in {11}:
            return k_mlp_0_1_output == 38
        elif q_mlp_0_1_output in {18, 12}:
            return k_mlp_0_1_output == 77
        elif q_mlp_0_1_output in {42, 74, 13, 78, 15, 19, 29, 62, 31}:
            return k_mlp_0_1_output == 11
        elif q_mlp_0_1_output in {33, 14, 79}:
            return k_mlp_0_1_output == 61
        elif q_mlp_0_1_output in {16, 58}:
            return k_mlp_0_1_output == 8
        elif q_mlp_0_1_output in {17, 35, 51}:
            return k_mlp_0_1_output == 20
        elif q_mlp_0_1_output in {20}:
            return k_mlp_0_1_output == 45
        elif q_mlp_0_1_output in {43, 27, 21}:
            return k_mlp_0_1_output == 46
        elif q_mlp_0_1_output in {72, 22}:
            return k_mlp_0_1_output == 70
        elif q_mlp_0_1_output in {60, 39, 70, 23}:
            return k_mlp_0_1_output == 49
        elif q_mlp_0_1_output in {24, 61}:
            return k_mlp_0_1_output == 36
        elif q_mlp_0_1_output in {32, 26, 38}:
            return k_mlp_0_1_output == 25
        elif q_mlp_0_1_output in {28}:
            return k_mlp_0_1_output == 10
        elif q_mlp_0_1_output in {77, 30}:
            return k_mlp_0_1_output == 59
        elif q_mlp_0_1_output in {34, 69}:
            return k_mlp_0_1_output == 28
        elif q_mlp_0_1_output in {36}:
            return k_mlp_0_1_output == 34
        elif q_mlp_0_1_output in {53, 37}:
            return k_mlp_0_1_output == 78
        elif q_mlp_0_1_output in {41}:
            return k_mlp_0_1_output == 67
        elif q_mlp_0_1_output in {46}:
            return k_mlp_0_1_output == 19
        elif q_mlp_0_1_output in {48}:
            return k_mlp_0_1_output == 30
        elif q_mlp_0_1_output in {49}:
            return k_mlp_0_1_output == 4
        elif q_mlp_0_1_output in {52}:
            return k_mlp_0_1_output == 16
        elif q_mlp_0_1_output in {57, 76}:
            return k_mlp_0_1_output == 37
        elif q_mlp_0_1_output in {65}:
            return k_mlp_0_1_output == 27
        elif q_mlp_0_1_output in {66}:
            return k_mlp_0_1_output == 56
        elif q_mlp_0_1_output in {67}:
            return k_mlp_0_1_output == 58
        elif q_mlp_0_1_output in {68}:
            return k_mlp_0_1_output == 23
        elif q_mlp_0_1_output in {71}:
            return k_mlp_0_1_output == 51
        elif q_mlp_0_1_output in {73}:
            return k_mlp_0_1_output == 47

    num_attn_1_6_pattern = select(mlp_0_1_outputs, mlp_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_3_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_4_output, attn_0_5_output):
        if attn_0_4_output in {"(", "<s>", ")"}:
            return attn_0_5_output == ""

    num_attn_1_7_pattern = select(attn_0_5_outputs, attn_0_4_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_7_output, attn_1_5_output):
        key = (attn_1_7_output, attn_1_5_output)
        if key in {("(", ")"), (")", "(")}:
            return 59
        elif key in {(")", ")"), ("<s>", ")")}:
            return 2
        elif key in {(")", "<s>")}:
            return 62
        return 36

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_7_outputs, attn_1_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_5_output, attn_0_4_output):
        key = (attn_0_5_output, attn_0_4_output)
        return 43

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_4_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_4_output, num_attn_1_1_output):
        key = (num_attn_1_4_output, num_attn_1_1_output)
        return 49

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_6_output, num_attn_1_5_output):
        key = (num_attn_1_6_output, num_attn_1_5_output)
        return 44

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_4_output, position):
        if attn_0_4_output in {"("}:
            return position == 4
        elif attn_0_4_output in {")"}:
            return position == 13
        elif attn_0_4_output in {"<s>"}:
            return position == 2

    attn_2_0_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_4_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_2_output, position):
        if attn_0_2_output in {"(", ")"}:
            return position == 11
        elif attn_0_2_output in {"<s>"}:
            return position == 1

    attn_2_1_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_4_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 3
        elif attn_0_2_output in {")"}:
            return position == 11
        elif attn_0_2_output in {"<s>"}:
            return position == 2

    attn_2_2_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, position):
        if token in {"("}:
            return position == 11
        elif token in {")"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 5

    attn_2_3_pattern = select_closest(positions, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_4_output, position):
        if attn_0_4_output in {"("}:
            return position == 6
        elif attn_0_4_output in {")"}:
            return position == 7
        elif attn_0_4_output in {"<s>"}:
            return position == 1

    attn_2_4_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_5_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 4
        elif attn_0_2_output in {")"}:
            return position == 5
        elif attn_0_2_output in {"<s>"}:
            return position == 1

    attn_2_5_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_0_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(token, position):
        if token in {"("}:
            return position == 2
        elif token in {")"}:
            return position == 11
        elif token in {"<s>"}:
            return position == 5

    attn_2_6_pattern = select_closest(positions, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_4_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(token, position):
        if token in {"(", ")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 5

    attn_2_7_pattern = select_closest(positions, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_4_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_1_1_output, num_mlp_0_0_output):
        if mlp_1_1_output in {0, 18}:
            return num_mlp_0_0_output == 49
        elif mlp_1_1_output in {1, 75, 12}:
            return num_mlp_0_0_output == 61
        elif mlp_1_1_output in {2, 58, 60}:
            return num_mlp_0_0_output == 40
        elif mlp_1_1_output in {3}:
            return num_mlp_0_0_output == 38
        elif mlp_1_1_output in {57, 43, 4, 45}:
            return num_mlp_0_0_output == 76
        elif mlp_1_1_output in {5, 38, 79}:
            return num_mlp_0_0_output == 55
        elif mlp_1_1_output in {6, 30}:
            return num_mlp_0_0_output == 48
        elif mlp_1_1_output in {7}:
            return num_mlp_0_0_output == 75
        elif mlp_1_1_output in {8, 35, 63}:
            return num_mlp_0_0_output == 46
        elif mlp_1_1_output in {9, 44}:
            return num_mlp_0_0_output == 18
        elif mlp_1_1_output in {10, 39}:
            return num_mlp_0_0_output == 51
        elif mlp_1_1_output in {11, 31}:
            return num_mlp_0_0_output == 68
        elif mlp_1_1_output in {13}:
            return num_mlp_0_0_output == 5
        elif mlp_1_1_output in {16, 14}:
            return num_mlp_0_0_output == 56
        elif mlp_1_1_output in {15}:
            return num_mlp_0_0_output == 53
        elif mlp_1_1_output in {17, 42}:
            return num_mlp_0_0_output == 4
        elif mlp_1_1_output in {19}:
            return num_mlp_0_0_output == 32
        elif mlp_1_1_output in {20, 29}:
            return num_mlp_0_0_output == 25
        elif mlp_1_1_output in {21}:
            return num_mlp_0_0_output == 8
        elif mlp_1_1_output in {22}:
            return num_mlp_0_0_output == 13
        elif mlp_1_1_output in {28, 23}:
            return num_mlp_0_0_output == 59
        elif mlp_1_1_output in {24}:
            return num_mlp_0_0_output == 52
        elif mlp_1_1_output in {25, 50, 46}:
            return num_mlp_0_0_output == 62
        elif mlp_1_1_output in {26, 37}:
            return num_mlp_0_0_output == 64
        elif mlp_1_1_output in {27}:
            return num_mlp_0_0_output == 79
        elif mlp_1_1_output in {32, 53}:
            return num_mlp_0_0_output == 66
        elif mlp_1_1_output in {33}:
            return num_mlp_0_0_output == 58
        elif mlp_1_1_output in {34, 59}:
            return num_mlp_0_0_output == 60
        elif mlp_1_1_output in {64, 36}:
            return num_mlp_0_0_output == 47
        elif mlp_1_1_output in {40}:
            return num_mlp_0_0_output == 44
        elif mlp_1_1_output in {41}:
            return num_mlp_0_0_output == 63
        elif mlp_1_1_output in {47}:
            return num_mlp_0_0_output == 50
        elif mlp_1_1_output in {48}:
            return num_mlp_0_0_output == 0
        elif mlp_1_1_output in {49}:
            return num_mlp_0_0_output == 15
        elif mlp_1_1_output in {51, 61}:
            return num_mlp_0_0_output == 70
        elif mlp_1_1_output in {52}:
            return num_mlp_0_0_output == 74
        elif mlp_1_1_output in {54}:
            return num_mlp_0_0_output == 22
        elif mlp_1_1_output in {55}:
            return num_mlp_0_0_output == 43
        elif mlp_1_1_output in {56}:
            return num_mlp_0_0_output == 71
        elif mlp_1_1_output in {76, 62}:
            return num_mlp_0_0_output == 67
        elif mlp_1_1_output in {65}:
            return num_mlp_0_0_output == 16
        elif mlp_1_1_output in {66}:
            return num_mlp_0_0_output == 69
        elif mlp_1_1_output in {67}:
            return num_mlp_0_0_output == 33
        elif mlp_1_1_output in {68}:
            return num_mlp_0_0_output == 23
        elif mlp_1_1_output in {69, 78}:
            return num_mlp_0_0_output == 2
        elif mlp_1_1_output in {70}:
            return num_mlp_0_0_output == 65
        elif mlp_1_1_output in {71}:
            return num_mlp_0_0_output == 78
        elif mlp_1_1_output in {72}:
            return num_mlp_0_0_output == 19
        elif mlp_1_1_output in {73}:
            return num_mlp_0_0_output == 41
        elif mlp_1_1_output in {74}:
            return num_mlp_0_0_output == 77
        elif mlp_1_1_output in {77}:
            return num_mlp_0_0_output == 36

    num_attn_2_0_pattern = select(
        num_mlp_0_0_outputs, mlp_1_1_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_1_1_output, mlp_0_1_output):
        if mlp_1_1_output in {0, 33, 73}:
            return mlp_0_1_output == 64
        elif mlp_1_1_output in {1}:
            return mlp_0_1_output == 68
        elif mlp_1_1_output in {2, 22}:
            return mlp_0_1_output == 20
        elif mlp_1_1_output in {3, 68, 39, 11, 13, 54, 24, 56, 30}:
            return mlp_0_1_output == 46
        elif mlp_1_1_output in {64, 4, 5, 47, 19, 26}:
            return mlp_0_1_output == 16
        elif mlp_1_1_output in {49, 59, 44, 6}:
            return mlp_0_1_output == 70
        elif mlp_1_1_output in {48, 7}:
            return mlp_0_1_output == 56
        elif mlp_1_1_output in {8, 9, 36, 65}:
            return mlp_0_1_output == 49
        elif mlp_1_1_output in {10, 12, 60, 62, 63}:
            return mlp_0_1_output == 25
        elif mlp_1_1_output in {34, 38, 14, 79, 50, 20, 25}:
            return mlp_0_1_output == 11
        elif mlp_1_1_output in {77, 15}:
            return mlp_0_1_output == 59
        elif mlp_1_1_output in {16, 75, 35, 31}:
            return mlp_0_1_output == 34
        elif mlp_1_1_output in {17}:
            return mlp_0_1_output == 24
        elif mlp_1_1_output in {18}:
            return mlp_0_1_output == 9
        elif mlp_1_1_output in {67, 37, 74, 76, 45, 21}:
            return mlp_0_1_output == 79
        elif mlp_1_1_output in {23}:
            return mlp_0_1_output == 37
        elif mlp_1_1_output in {27}:
            return mlp_0_1_output == 55
        elif mlp_1_1_output in {32, 66, 72, 28}:
            return mlp_0_1_output == 39
        elif mlp_1_1_output in {29}:
            return mlp_0_1_output == 52
        elif mlp_1_1_output in {40, 41, 52, 55}:
            return mlp_0_1_output == 18
        elif mlp_1_1_output in {70, 42, 43, 46}:
            return mlp_0_1_output == 13
        elif mlp_1_1_output in {51, 53}:
            return mlp_0_1_output == 76
        elif mlp_1_1_output in {57, 61}:
            return mlp_0_1_output == 1
        elif mlp_1_1_output in {58}:
            return mlp_0_1_output == 77
        elif mlp_1_1_output in {69}:
            return mlp_0_1_output == 51
        elif mlp_1_1_output in {78, 71}:
            return mlp_0_1_output == 40

    num_attn_2_1_pattern = select(mlp_0_1_outputs, mlp_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_7_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(num_mlp_1_1_output, mlp_1_0_output):
        if num_mlp_1_1_output in {
            0,
            65,
            34,
            35,
            36,
            5,
            37,
            38,
            40,
            66,
            73,
            75,
            44,
            47,
            49,
            63,
        }:
            return mlp_1_0_output == 29
        elif num_mlp_1_1_output in {1, 33, 10, 55, 60}:
            return mlp_1_0_output == 14
        elif num_mlp_1_1_output in {2, 4, 12, 76, 77, 78, 20, 57}:
            return mlp_1_0_output == 19
        elif num_mlp_1_1_output in {3, 41, 45, 13, 48, 59}:
            return mlp_1_0_output == 35
        elif num_mlp_1_1_output in {62, 43, 6}:
            return mlp_1_0_output == 56
        elif num_mlp_1_1_output in {53, 7}:
            return mlp_1_0_output == 69
        elif num_mlp_1_1_output in {8, 16}:
            return mlp_1_0_output == 76
        elif num_mlp_1_1_output in {9, 46, 70}:
            return mlp_1_0_output == 33
        elif num_mlp_1_1_output in {11, 15}:
            return mlp_1_0_output == 22
        elif num_mlp_1_1_output in {74, 14}:
            return mlp_1_0_output == 52
        elif num_mlp_1_1_output in {17, 50, 19, 22, 26}:
            return mlp_1_0_output == 24
        elif num_mlp_1_1_output in {18}:
            return mlp_1_0_output == 25
        elif num_mlp_1_1_output in {24, 61, 21, 23}:
            return mlp_1_0_output == 18
        elif num_mlp_1_1_output in {25, 42, 52}:
            return mlp_1_0_output == 79
        elif num_mlp_1_1_output in {27, 31}:
            return mlp_1_0_output == 49
        elif num_mlp_1_1_output in {28}:
            return mlp_1_0_output == 9
        elif num_mlp_1_1_output in {69, 29, 54}:
            return mlp_1_0_output == 27
        elif num_mlp_1_1_output in {30}:
            return mlp_1_0_output == 16
        elif num_mlp_1_1_output in {32}:
            return mlp_1_0_output == 11
        elif num_mlp_1_1_output in {39}:
            return mlp_1_0_output == 58
        elif num_mlp_1_1_output in {51}:
            return mlp_1_0_output == 61
        elif num_mlp_1_1_output in {56}:
            return mlp_1_0_output == 21
        elif num_mlp_1_1_output in {58}:
            return mlp_1_0_output == 7
        elif num_mlp_1_1_output in {64}:
            return mlp_1_0_output == 30
        elif num_mlp_1_1_output in {67}:
            return mlp_1_0_output == 74
        elif num_mlp_1_1_output in {68}:
            return mlp_1_0_output == 47
        elif num_mlp_1_1_output in {71}:
            return mlp_1_0_output == 54
        elif num_mlp_1_1_output in {72}:
            return mlp_1_0_output == 15
        elif num_mlp_1_1_output in {79}:
            return mlp_1_0_output == 43

    num_attn_2_2_pattern = select(
        mlp_1_0_outputs, num_mlp_1_1_outputs, num_predicate_2_2
    )
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_7_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_4_output, mlp_0_1_output):
        if attn_1_4_output in {"("}:
            return mlp_0_1_output == 56
        elif attn_1_4_output in {")"}:
            return mlp_0_1_output == 25
        elif attn_1_4_output in {"<s>"}:
            return mlp_0_1_output == 46

    num_attn_2_3_pattern = select(mlp_0_1_outputs, attn_1_4_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(mlp_1_1_output, mlp_1_0_output):
        if mlp_1_1_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            9,
            11,
            12,
            13,
            15,
            16,
            19,
            21,
            24,
            25,
            33,
            35,
            37,
            38,
            41,
            44,
            45,
            46,
            47,
            49,
            51,
            52,
            53,
            55,
            56,
            58,
            59,
            61,
            66,
            67,
            68,
            69,
            70,
            71,
            73,
            74,
            75,
            79,
        }:
            return mlp_1_0_output == 2
        elif mlp_1_1_output in {40, 7}:
            return mlp_1_0_output == 75
        elif mlp_1_1_output in {
            65,
            8,
            42,
            43,
            77,
            14,
            60,
            48,
            50,
            20,
            22,
            57,
            28,
            29,
            63,
        }:
            return mlp_1_0_output == 41
        elif mlp_1_1_output in {39, 10, 76, 17, 62, 30, 31}:
            return mlp_1_0_output == 23
        elif mlp_1_1_output in {34, 78, 18, 54, 26, 27}:
            return mlp_1_0_output == 34
        elif mlp_1_1_output in {23}:
            return mlp_1_0_output == 17
        elif mlp_1_1_output in {32}:
            return mlp_1_0_output == 28
        elif mlp_1_1_output in {36}:
            return mlp_1_0_output == 40
        elif mlp_1_1_output in {64}:
            return mlp_1_0_output == 42
        elif mlp_1_1_output in {72}:
            return mlp_1_0_output == 48

    num_attn_2_4_pattern = select(mlp_1_0_outputs, mlp_1_1_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_6_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(mlp_1_0_output, position):
        if mlp_1_0_output in {0, 73, 44}:
            return position == 24
        elif mlp_1_0_output in {1, 34, 4, 37, 11, 48, 55}:
            return position == 37
        elif mlp_1_0_output in {
            65,
            2,
            3,
            36,
            67,
            38,
            39,
            9,
            41,
            13,
            77,
            49,
            18,
            62,
            59,
            60,
            30,
            31,
        }:
            return position == 30
        elif mlp_1_0_output in {5, 72, 42, 45, 23, 25}:
            return position == 34
        elif mlp_1_0_output in {58, 6}:
            return position == 33
        elif mlp_1_0_output in {7, 10, 74, 12, 75, 76, 15, 57}:
            return position == 26
        elif mlp_1_0_output in {64, 35, 68, 8, 19, 52, 61}:
            return position == 36
        elif mlp_1_0_output in {14, 63}:
            return position == 25
        elif mlp_1_0_output in {16, 43, 79}:
            return position == 35
        elif mlp_1_0_output in {69, 40, 46, 47, 78, 17, 27}:
            return position == 32
        elif mlp_1_0_output in {20}:
            return position == 57
        elif mlp_1_0_output in {21}:
            return position == 59
        elif mlp_1_0_output in {26, 51, 22}:
            return position == 28
        elif mlp_1_0_output in {24}:
            return position == 20
        elif mlp_1_0_output in {56, 50, 28, 70}:
            return position == 31
        elif mlp_1_0_output in {29}:
            return position == 60
        elif mlp_1_0_output in {32, 66, 54}:
            return position == 29
        elif mlp_1_0_output in {33}:
            return position == 22
        elif mlp_1_0_output in {53}:
            return position == 42
        elif mlp_1_0_output in {71}:
            return position == 68

    num_attn_2_5_pattern = select(positions, mlp_1_0_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_3_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0, 65, 2, 42, 43, 17, 21, 55}:
            return mlp_0_1_output == 64
        elif mlp_0_0_output in {16, 1, 78}:
            return mlp_0_1_output == 25
        elif mlp_0_0_output in {50, 3, 79}:
            return mlp_0_1_output == 13
        elif mlp_0_0_output in {73, 4}:
            return mlp_0_1_output == 59
        elif mlp_0_0_output in {67, 5, 74, 15, 48, 19, 52, 57, 26, 60, 62, 31}:
            return mlp_0_1_output == 1
        elif mlp_0_0_output in {40, 6}:
            return mlp_0_1_output == 61
        elif mlp_0_0_output in {38, 7, 8, 41, 72, 13, 47, 61, 63}:
            return mlp_0_1_output == 37
        elif mlp_0_0_output in {9}:
            return mlp_0_1_output == 36
        elif mlp_0_0_output in {36, 69, 10, 77, 45}:
            return mlp_0_1_output == 11
        elif mlp_0_0_output in {32, 34, 11, 75, 59}:
            return mlp_0_1_output == 16
        elif mlp_0_0_output in {12}:
            return mlp_0_1_output == 56
        elif mlp_0_0_output in {14, 71}:
            return mlp_0_1_output == 69
        elif mlp_0_0_output in {18}:
            return mlp_0_1_output == 79
        elif mlp_0_0_output in {25, 20}:
            return mlp_0_1_output == 78
        elif mlp_0_0_output in {22}:
            return mlp_0_1_output == 68
        elif mlp_0_0_output in {35, 37, 23}:
            return mlp_0_1_output == 18
        elif mlp_0_0_output in {24}:
            return mlp_0_1_output == 40
        elif mlp_0_0_output in {27}:
            return mlp_0_1_output == 39
        elif mlp_0_0_output in {28, 54}:
            return mlp_0_1_output == 70
        elif mlp_0_0_output in {49, 29, 46, 39}:
            return mlp_0_1_output == 30
        elif mlp_0_0_output in {30}:
            return mlp_0_1_output == 24
        elif mlp_0_0_output in {33}:
            return mlp_0_1_output == 77
        elif mlp_0_0_output in {44}:
            return mlp_0_1_output == 46
        elif mlp_0_0_output in {51}:
            return mlp_0_1_output == 7
        elif mlp_0_0_output in {53}:
            return mlp_0_1_output == 32
        elif mlp_0_0_output in {56, 68}:
            return mlp_0_1_output == 22
        elif mlp_0_0_output in {58}:
            return mlp_0_1_output == 67
        elif mlp_0_0_output in {64}:
            return mlp_0_1_output == 34
        elif mlp_0_0_output in {66}:
            return mlp_0_1_output == 26
        elif mlp_0_0_output in {70}:
            return mlp_0_1_output == 49
        elif mlp_0_0_output in {76}:
            return mlp_0_1_output == 19

    num_attn_2_6_pattern = select(mlp_0_1_outputs, mlp_0_0_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_3_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_7_output, mlp_0_1_output):
        if attn_1_7_output in {"("}:
            return mlp_0_1_output == 45
        elif attn_1_7_output in {")"}:
            return mlp_0_1_output == 16
        elif attn_1_7_output in {"<s>"}:
            return mlp_0_1_output == 79

    num_attn_2_7_pattern = select(mlp_0_1_outputs, attn_1_7_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_7_output):
        key = attn_2_7_output
        if key in {")"}:
            return 10
        return 71

    mlp_2_0_outputs = [mlp_2_0(k0) for k0 in attn_2_7_outputs]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_4_output, attn_2_7_output):
        key = (attn_2_4_output, attn_2_7_output)
        if key in {(")", "<s>")}:
            return 0
        elif key in {("(", ")"), ("<s>", ")")}:
            return 66
        elif key in {(")", ")")}:
            return 1
        return 25

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_4_outputs, attn_2_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_1_output, num_attn_1_0_output):
        key = (num_attn_1_1_output, num_attn_1_0_output)
        return 35

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_7_output, num_attn_1_0_output):
        key = (num_attn_1_7_output, num_attn_1_0_output)
        return 56

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
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
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_0_4_output_scores,
                num_attn_0_5_output_scores,
                num_attn_0_6_output_scores,
                num_attn_0_7_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_1_4_output_scores,
                num_attn_1_5_output_scores,
                num_attn_1_6_output_scores,
                num_attn_1_7_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_2_4_output_scores,
                num_attn_2_5_output_scores,
                num_attn_2_6_output_scores,
                num_attn_2_7_output_scores,
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
            ")",
            ")",
            "(",
            "(",
            ")",
            "(",
            ")",
            "(",
            ")",
            "(",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            "(",
            "(",
            "(",
            "(",
            ")",
            ")",
            ")",
            "(",
            "(",
            "(",
            ")",
            ")",
            ")",
            "(",
            "(",
            ")",
            "(",
            "(",
            ")",
            ")",
            ")",
            "(",
        ]
    )
)
