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
        "output/length/rasp/dyck1/trainlength40/s3/dyck1_weights.csv",
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
    def predicate_0_0(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 56

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 49
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 4}:
            return k_position == 2
        elif q_position in {3, 5}:
            return k_position == 3
        elif q_position in {33, 67, 68, 37, 6, 7, 8, 39, 51}:
            return k_position == 5
        elif q_position in {70, 40, 9, 10, 47, 55, 59}:
            return k_position == 7
        elif q_position in {64, 11, 76, 54, 30}:
            return k_position == 9
        elif q_position in {17, 12, 14}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 8
        elif q_position in {28, 15}:
            return k_position == 10
        elif q_position in {16, 57, 60}:
            return k_position == 15
        elif q_position in {18, 42}:
            return k_position == 17
        elif q_position in {24, 19, 21, 23}:
            return k_position == 16
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {25, 62}:
            return k_position == 22
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27, 44, 29}:
            return k_position == 24
        elif q_position in {31}:
            return k_position == 18
        elif q_position in {32}:
            return k_position == 27
        elif q_position in {34}:
            return k_position == 12
        elif q_position in {35}:
            return k_position == 4
        elif q_position in {36}:
            return k_position == 34
        elif q_position in {38}:
            return k_position == 36
        elif q_position in {41, 43, 77}:
            return k_position == 46
        elif q_position in {45}:
            return k_position == 54
        elif q_position in {46}:
            return k_position == 67
        elif q_position in {48}:
            return k_position == 57
        elif q_position in {49}:
            return k_position == 70
        elif q_position in {50}:
            return k_position == 73
        elif q_position in {52}:
            return k_position == 71
        elif q_position in {53, 63}:
            return k_position == 64
        elif q_position in {56}:
            return k_position == 58
        elif q_position in {58, 79}:
            return k_position == 55
        elif q_position in {72, 61}:
            return k_position == 78
        elif q_position in {65}:
            return k_position == 33
        elif q_position in {66}:
            return k_position == 29
        elif q_position in {69}:
            return k_position == 37
        elif q_position in {71}:
            return k_position == 60
        elif q_position in {73}:
            return k_position == 26
        elif q_position in {74}:
            return k_position == 76
        elif q_position in {75}:
            return k_position == 40
        elif q_position in {78}:
            return k_position == 51

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {
            0,
            8,
            9,
            11,
            13,
            17,
            24,
            26,
            28,
            30,
            39,
            41,
            42,
            43,
            47,
            48,
            51,
            55,
            63,
            70,
            74,
            77,
        }:
            return k_position == 7
        elif q_position in {1, 3}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {4, 5}:
            return k_position == 3
        elif q_position in {67, 6, 7, 57, 58, 59}:
            return k_position == 5
        elif q_position in {10, 18, 12}:
            return k_position == 8
        elif q_position in {34, 14}:
            return k_position == 12
        elif q_position in {32, 66, 68, 46, 15, 16, 52, 60}:
            return k_position == 9
        elif q_position in {65, 19, 36, 79}:
            return k_position == 14
        elif q_position in {25, 20}:
            return k_position == 15
        elif q_position in {21, 22}:
            return k_position == 16
        elif q_position in {29, 23}:
            return k_position == 18
        elif q_position in {27}:
            return k_position == 21
        elif q_position in {37, 31}:
            return k_position == 4
        elif q_position in {33, 35}:
            return k_position == 6
        elif q_position in {38}:
            return k_position == 37
        elif q_position in {40}:
            return k_position == 65
        elif q_position in {44}:
            return k_position == 72
        elif q_position in {45}:
            return k_position == 57
        elif q_position in {49}:
            return k_position == 55
        elif q_position in {50}:
            return k_position == 71
        elif q_position in {53}:
            return k_position == 61
        elif q_position in {56, 54}:
            return k_position == 22
        elif q_position in {61}:
            return k_position == 67
        elif q_position in {62}:
            return k_position == 31
        elif q_position in {64}:
            return k_position == 66
        elif q_position in {69}:
            return k_position == 45
        elif q_position in {78, 71}:
            return k_position == 52
        elif q_position in {72}:
            return k_position == 63
        elif q_position in {73}:
            return k_position == 70
        elif q_position in {75}:
            return k_position == 73
        elif q_position in {76}:
            return k_position == 39

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 74
        elif q_position in {1, 3, 4}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 37
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8, 39, 37, 7}:
            return k_position == 6
        elif q_position in {9, 13}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {34, 12}:
            return k_position == 11
        elif q_position in {42, 14, 55}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20, 22}:
            return k_position == 19
        elif q_position in {27, 21, 23}:
            return k_position == 20
        elif q_position in {24}:
            return k_position == 12
        elif q_position in {25, 61}:
            return k_position == 24
        elif q_position in {26, 47}:
            return k_position == 25
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {58, 29}:
            return k_position == 28
        elif q_position in {32, 30}:
            return k_position == 29
        elif q_position in {41, 78, 31}:
            return k_position == 30
        elif q_position in {33}:
            return k_position == 32
        elif q_position in {74, 35, 38}:
            return k_position == 34
        elif q_position in {36}:
            return k_position == 35
        elif q_position in {40}:
            return k_position == 62
        elif q_position in {70, 71, 72, 43, 44, 75, 51}:
            return k_position == 26
        elif q_position in {65, 45, 46}:
            return k_position == 22
        elif q_position in {48}:
            return k_position == 76
        elif q_position in {49}:
            return k_position == 64
        elif q_position in {50}:
            return k_position == 73
        elif q_position in {52}:
            return k_position == 77
        elif q_position in {64, 53}:
            return k_position == 42
        elif q_position in {54}:
            return k_position == 23
        elif q_position in {56}:
            return k_position == 58
        elif q_position in {57}:
            return k_position == 40
        elif q_position in {59}:
            return k_position == 63
        elif q_position in {60}:
            return k_position == 65
        elif q_position in {62}:
            return k_position == 46
        elif q_position in {63}:
            return k_position == 44
        elif q_position in {66, 79}:
            return k_position == 79
        elif q_position in {67}:
            return k_position == 53
        elif q_position in {68}:
            return k_position == 61
        elif q_position in {69}:
            return k_position == 70
        elif q_position in {73}:
            return k_position == 57
        elif q_position in {76}:
            return k_position == 55
        elif q_position in {77}:
            return k_position == 39

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
            return position == 4
        elif token in {"<s>"}:
            return position == 3

    attn_0_4_pattern = select_closest(positions, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {
            0,
            5,
            12,
            16,
            20,
            21,
            23,
            24,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            35,
            37,
            39,
            43,
            45,
            46,
            50,
            53,
            56,
            59,
            60,
            63,
            65,
            67,
            68,
            78,
        }:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 11, 38}:
            return k_position == 38
        elif q_position in {75, 3, 71}:
            return k_position == 44
        elif q_position in {34, 4, 36, 79}:
            return k_position == 11
        elif q_position in {6}:
            return k_position == 13
        elif q_position in {22, 7}:
            return k_position == 49
        elif q_position in {8, 25, 41}:
            return k_position == 5
        elif q_position in {9, 44, 14}:
            return k_position == 78
        elif q_position in {10}:
            return k_position == 37
        elif q_position in {13}:
            return k_position == 50
        elif q_position in {15}:
            return k_position == 60
        elif q_position in {17}:
            return k_position == 62
        elif q_position in {18, 76}:
            return k_position == 55
        elif q_position in {19, 62, 47}:
            return k_position == 74
        elif q_position in {40, 77}:
            return k_position == 56
        elif q_position in {42}:
            return k_position == 51
        elif q_position in {48, 54}:
            return k_position == 46
        elif q_position in {49, 70}:
            return k_position == 77
        elif q_position in {51, 55}:
            return k_position == 9
        elif q_position in {52}:
            return k_position == 72
        elif q_position in {57}:
            return k_position == 41
        elif q_position in {58, 69}:
            return k_position == 76
        elif q_position in {64, 74, 72, 61}:
            return k_position == 39
        elif q_position in {66}:
            return k_position == 58
        elif q_position in {73}:
            return k_position == 57

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 3

    attn_0_6_pattern = select_closest(positions, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 3, 41, 47, 50}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 51
        elif q_position in {2, 4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 52
        elif q_position in {33, 37, 6, 7, 39, 9}:
            return k_position == 4
        elif q_position in {
            64,
            67,
            58,
            69,
            70,
            8,
            42,
            75,
            77,
            78,
            55,
            57,
            26,
            60,
            62,
            63,
        }:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {25, 11, 13}:
            return k_position == 6
        elif q_position in {24, 12, 53}:
            return k_position == 8
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {38, 15}:
            return k_position == 11
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17, 68}:
            return k_position == 14
        elif q_position in {18, 45}:
            return k_position == 17
        elif q_position in {19, 31}:
            return k_position == 18
        elif q_position in {73, 20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 10
        elif q_position in {74, 22}:
            return k_position == 21
        elif q_position in {28, 23}:
            return k_position == 22
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {29}:
            return k_position == 23
        elif q_position in {30}:
            return k_position == 27
        elif q_position in {32, 36}:
            return k_position == 31
        elif q_position in {34, 46}:
            return k_position == 33
        elif q_position in {35}:
            return k_position == 34
        elif q_position in {40}:
            return k_position == 72
        elif q_position in {43, 54, 71}:
            return k_position == 12
        elif q_position in {49, 59, 44}:
            return k_position == 25
        elif q_position in {48, 56}:
            return k_position == 45
        elif q_position in {51}:
            return k_position == 3
        elif q_position in {52}:
            return k_position == 69
        elif q_position in {61}:
            return k_position == 55
        elif q_position in {65}:
            return k_position == 63
        elif q_position in {66}:
            return k_position == 53
        elif q_position in {72}:
            return k_position == 76
        elif q_position in {76}:
            return k_position == 35
        elif q_position in {79}:
            return k_position == 78

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 33, 5}:
            return k_position == 45
        elif q_position in {1, 6}:
            return k_position == 76
        elif q_position in {2}:
            return k_position == 54
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {24, 4, 23}:
            return k_position == 32
        elif q_position in {64, 53, 7}:
            return k_position == 42
        elif q_position in {8}:
            return k_position == 27
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 6
        elif q_position in {11}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 26
        elif q_position in {13}:
            return k_position == 3
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 28
        elif q_position in {17, 62}:
            return k_position == 58
        elif q_position in {65, 18}:
            return k_position == 53
        elif q_position in {26, 19, 70}:
            return k_position == 36
        elif q_position in {67, 20}:
            return k_position == 50
        elif q_position in {21}:
            return k_position == 40
        elif q_position in {22}:
            return k_position == 65
        elif q_position in {25, 35, 77}:
            return k_position == 37
        elif q_position in {72, 27}:
            return k_position == 35
        elif q_position in {28, 37, 78}:
            return k_position == 56
        elif q_position in {40, 29}:
            return k_position == 75
        elif q_position in {57, 30, 63}:
            return k_position == 77
        elif q_position in {31}:
            return k_position == 68
        elif q_position in {32}:
            return k_position == 55
        elif q_position in {34, 43}:
            return k_position == 69
        elif q_position in {36}:
            return k_position == 57
        elif q_position in {38}:
            return k_position == 78
        elif q_position in {51, 39}:
            return k_position == 47
        elif q_position in {41, 49}:
            return k_position == 72
        elif q_position in {42, 44, 45, 54}:
            return k_position == 41
        elif q_position in {46, 79}:
            return k_position == 12
        elif q_position in {48, 66, 47}:
            return k_position == 59
        elif q_position in {50, 55}:
            return k_position == 71
        elif q_position in {52, 69, 68}:
            return k_position == 52
        elif q_position in {56}:
            return k_position == 46
        elif q_position in {58}:
            return k_position == 21
        elif q_position in {59, 76}:
            return k_position == 44
        elif q_position in {60, 61, 71}:
            return k_position == 61
        elif q_position in {73}:
            return k_position == 11
        elif q_position in {74}:
            return k_position == 34
        elif q_position in {75}:
            return k_position == 70

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 57}:
            return k_position == 68
        elif q_position in {1, 50}:
            return k_position == 78
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {3}:
            return k_position == 75
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {73, 5}:
            return k_position == 52
        elif q_position in {10, 6}:
            return k_position == 11
        elif q_position in {64, 20, 7}:
            return k_position == 47
        elif q_position in {8, 17}:
            return k_position == 21
        elif q_position in {9}:
            return k_position == 3
        elif q_position in {11, 79}:
            return k_position == 62
        elif q_position in {12}:
            return k_position == 1
        elif q_position in {13}:
            return k_position == 5
        elif q_position in {44, 14}:
            return k_position == 76
        elif q_position in {15}:
            return k_position == 22
        elif q_position in {16}:
            return k_position == 26
        elif q_position in {18, 31, 23}:
            return k_position == 36
        elif q_position in {19}:
            return k_position == 32
        elif q_position in {51, 60, 21}:
            return k_position == 51
        elif q_position in {38, 62, 22}:
            return k_position == 42
        elif q_position in {24, 41, 65}:
            return k_position == 39
        elif q_position in {25, 39}:
            return k_position == 69
        elif q_position in {32, 26, 59}:
            return k_position == 41
        elif q_position in {27, 52}:
            return k_position == 73
        elif q_position in {34, 28, 71}:
            return k_position == 45
        elif q_position in {46, 29, 30}:
            return k_position == 72
        elif q_position in {33, 45, 70}:
            return k_position == 40
        elif q_position in {35, 61}:
            return k_position == 56
        elif q_position in {36}:
            return k_position == 79
        elif q_position in {66, 43, 37}:
            return k_position == 64
        elif q_position in {40, 56}:
            return k_position == 59
        elif q_position in {42}:
            return k_position == 54
        elif q_position in {47}:
            return k_position == 50
        elif q_position in {48}:
            return k_position == 43
        elif q_position in {49, 78}:
            return k_position == 60
        elif q_position in {53}:
            return k_position == 46
        elif q_position in {68, 54}:
            return k_position == 71
        elif q_position in {74, 55}:
            return k_position == 55
        elif q_position in {58}:
            return k_position == 74
        elif q_position in {77, 63}:
            return k_position == 77
        elif q_position in {67}:
            return k_position == 65
        elif q_position in {69}:
            return k_position == 37
        elif q_position in {72}:
            return k_position == 44
        elif q_position in {75}:
            return k_position == 67
        elif q_position in {76}:
            return k_position == 38

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 64, 44}:
            return k_position == 42
        elif q_position in {1, 60, 17}:
            return k_position == 36
        elif q_position in {2}:
            return k_position == 15
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4, 15}:
            return k_position == 59
        elif q_position in {13, 5}:
            return k_position == 66
        elif q_position in {48, 69, 6, 71}:
            return k_position == 79
        elif q_position in {7}:
            return k_position == 24
        elif q_position in {8, 68}:
            return k_position == 48
        elif q_position in {9}:
            return k_position == 23
        elif q_position in {16, 10}:
            return k_position == 47
        elif q_position in {74, 11, 22}:
            return k_position == 56
        elif q_position in {12}:
            return k_position == 72
        elif q_position in {38, 14}:
            return k_position == 43
        elif q_position in {18}:
            return k_position == 27
        elif q_position in {49, 75, 19, 77}:
            return k_position == 67
        elif q_position in {20, 76}:
            return k_position == 74
        elif q_position in {59, 21}:
            return k_position == 62
        elif q_position in {23}:
            return k_position == 28
        elif q_position in {24}:
            return k_position == 30
        elif q_position in {25, 46, 63}:
            return k_position == 68
        elif q_position in {32, 26, 70}:
            return k_position == 52
        elif q_position in {27}:
            return k_position == 64
        elif q_position in {28}:
            return k_position == 78
        elif q_position in {29, 47}:
            return k_position == 54
        elif q_position in {30}:
            return k_position == 34
        elif q_position in {31}:
            return k_position == 32
        elif q_position in {33}:
            return k_position == 76
        elif q_position in {34}:
            return k_position == 35
        elif q_position in {50, 35}:
            return k_position == 38
        elif q_position in {41, 36}:
            return k_position == 61
        elif q_position in {42, 53, 37}:
            return k_position == 73
        elif q_position in {45, 39}:
            return k_position == 45
        elif q_position in {40, 55}:
            return k_position == 44
        elif q_position in {72, 43}:
            return k_position == 51
        elif q_position in {51}:
            return k_position == 57
        elif q_position in {52}:
            return k_position == 41
        elif q_position in {54}:
            return k_position == 75
        elif q_position in {56}:
            return k_position == 37
        elif q_position in {57}:
            return k_position == 25
        elif q_position in {58}:
            return k_position == 39
        elif q_position in {61}:
            return k_position == 12
        elif q_position in {62}:
            return k_position == 77
        elif q_position in {65}:
            return k_position == 9
        elif q_position in {66}:
            return k_position == 46
        elif q_position in {67}:
            return k_position == 50
        elif q_position in {73}:
            return k_position == 40
        elif q_position in {78}:
            return k_position == 69
        elif q_position in {79}:
            return k_position == 22

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 10, 64}:
            return k_position == 67
        elif q_position in {1, 2, 51}:
            return k_position == 0
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 49
        elif q_position in {5}:
            return k_position == 62
        elif q_position in {19, 6}:
            return k_position == 56
        elif q_position in {35, 70, 7, 47, 49, 60}:
            return k_position == 72
        elif q_position in {8, 37}:
            return k_position == 60
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {11, 15}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {16, 65, 72, 13}:
            return k_position == 14
        elif q_position in {67, 38, 77, 14, 55, 63}:
            return k_position == 77
        elif q_position in {17, 76, 39}:
            return k_position == 45
        elif q_position in {18, 23}:
            return k_position == 30
        elif q_position in {20}:
            return k_position == 48
        elif q_position in {21}:
            return k_position == 18
        elif q_position in {56, 22}:
            return k_position == 53
        elif q_position in {24}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 69
        elif q_position in {26}:
            return k_position == 64
        elif q_position in {78, 27, 46, 71}:
            return k_position == 42
        elif q_position in {28, 68}:
            return k_position == 41
        elif q_position in {75, 29}:
            return k_position == 58
        elif q_position in {30}:
            return k_position == 75
        elif q_position in {31}:
            return k_position == 50
        elif q_position in {32, 41}:
            return k_position == 65
        elif q_position in {33}:
            return k_position == 66
        elif q_position in {34, 58}:
            return k_position == 39
        elif q_position in {36}:
            return k_position == 74
        elif q_position in {40}:
            return k_position == 40
        elif q_position in {42, 52}:
            return k_position == 76
        elif q_position in {43}:
            return k_position == 44
        elif q_position in {44}:
            return k_position == 78
        elif q_position in {45, 54}:
            return k_position == 63
        elif q_position in {48}:
            return k_position == 15
        elif q_position in {50}:
            return k_position == 52
        elif q_position in {53}:
            return k_position == 38
        elif q_position in {57}:
            return k_position == 21
        elif q_position in {66, 59}:
            return k_position == 70
        elif q_position in {61}:
            return k_position == 22
        elif q_position in {62}:
            return k_position == 54
        elif q_position in {69}:
            return k_position == 55
        elif q_position in {73}:
            return k_position == 71
        elif q_position in {74}:
            return k_position == 57
        elif q_position in {79}:
            return k_position == 26

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 67
        elif q_position in {24, 1}:
            return k_position == 59
        elif q_position in {2}:
            return k_position == 60
        elif q_position in {3}:
            return k_position == 23
        elif q_position in {4}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 11
        elif q_position in {67, 6}:
            return k_position == 77
        elif q_position in {36, 14, 7}:
            return k_position == 57
        elif q_position in {8, 29}:
            return k_position == 70
        elif q_position in {48, 9, 63}:
            return k_position == 76
        elif q_position in {10}:
            return k_position == 7
        elif q_position in {11}:
            return k_position == 52
        elif q_position in {25, 12, 78, 68}:
            return k_position == 37
        elif q_position in {42, 13}:
            return k_position == 27
        elif q_position in {60, 15}:
            return k_position == 55
        elif q_position in {16}:
            return k_position == 24
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {18, 28, 23}:
            return k_position == 47
        elif q_position in {26, 19}:
            return k_position == 50
        elif q_position in {32, 45, 20, 53, 54, 57}:
            return k_position == 36
        elif q_position in {21}:
            return k_position == 62
        elif q_position in {22}:
            return k_position == 39
        elif q_position in {34, 27}:
            return k_position == 65
        elif q_position in {50, 35, 30, 39}:
            return k_position == 74
        elif q_position in {31}:
            return k_position == 44
        elif q_position in {33}:
            return k_position == 58
        elif q_position in {37, 70}:
            return k_position == 42
        elif q_position in {38}:
            return k_position == 72
        elif q_position in {40, 59, 44, 76}:
            return k_position == 33
        elif q_position in {65, 41, 74, 75}:
            return k_position == 30
        elif q_position in {43, 46}:
            return k_position == 38
        elif q_position in {47}:
            return k_position == 73
        elif q_position in {72, 49, 55}:
            return k_position == 41
        elif q_position in {51}:
            return k_position == 45
        elif q_position in {56, 52}:
            return k_position == 31
        elif q_position in {64, 58, 62}:
            return k_position == 32
        elif q_position in {61}:
            return k_position == 48
        elif q_position in {66}:
            return k_position == 35
        elif q_position in {69}:
            return k_position == 29
        elif q_position in {71}:
            return k_position == 34
        elif q_position in {73}:
            return k_position == 66
        elif q_position in {77}:
            return k_position == 61
        elif q_position in {79}:
            return k_position == 78

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_token, k_token):
        if q_token in {"(", ")"}:
            return k_token == ""
        elif q_token in {"<s>"}:
            return k_token == ")"

    num_attn_0_5_pattern = select(tokens, tokens, num_predicate_0_5)
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
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            45,
            48,
            61,
            62,
            65,
            72,
            74,
            78,
            79,
        }:
            return token == ""
        elif position in {1, 2, 70}:
            return token == "<s>"
        elif position in {
            3,
            4,
            5,
            7,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
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
            63,
            64,
            66,
            67,
            68,
            69,
            71,
            73,
            75,
            76,
            77,
        }:
            return token == ")"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"("}:
            return position == 38
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 3

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_6_output, attn_0_3_output):
        key = (attn_0_6_output, attn_0_3_output)
        if key in {("(", "("), ("(", "<s>"), (")", "("), ("<s>", "(")}:
            return 1
        return 39

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_5_output, attn_0_3_output):
        key = (attn_0_5_output, attn_0_3_output)
        return 68

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_3_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_6_output, num_attn_0_7_output):
        key = (num_attn_0_6_output, num_attn_0_7_output)
        return 29

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_4_output, num_attn_0_1_output):
        key = (num_attn_0_4_output, num_attn_0_1_output)
        return 20

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_5_output, attn_0_4_output):
        if attn_0_5_output in {"("}:
            return attn_0_4_output == ""
        elif attn_0_5_output in {"<s>", ")"}:
            return attn_0_4_output == ")"

    attn_1_0_pattern = select_closest(attn_0_4_outputs, attn_0_5_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_4_output, position):
        if attn_0_4_output in {"("}:
            return position == 3
        elif attn_0_4_output in {")"}:
            return position == 58
        elif attn_0_4_output in {"<s>"}:
            return position == 1

    attn_1_1_pattern = select_closest(positions, attn_0_4_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_4_output, attn_0_5_output):
        if attn_0_4_output in {"("}:
            return attn_0_5_output == ""
        elif attn_0_4_output in {")"}:
            return attn_0_5_output == ")"
        elif attn_0_4_output in {"<s>"}:
            return attn_0_5_output == "("

    attn_1_2_pattern = select_closest(attn_0_5_outputs, attn_0_4_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"(", "<s>"}:
            return position == 1
        elif token in {")"}:
            return position == 7

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(token, position):
        if token in {"("}:
            return position == 3
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 56

    attn_1_4_pattern = select_closest(positions, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_4_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 3

    attn_1_5_pattern = select_closest(positions, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_5_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 3
        elif attn_0_2_output in {")"}:
            return position == 11
        elif attn_0_2_output in {"<s>"}:
            return position == 1

    attn_1_6_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_4_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 1
        elif attn_0_0_output in {")"}:
            return position == 5
        elif attn_0_0_output in {"<s>"}:
            return position == 3

    attn_1_7_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_5_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_7_output, attn_0_0_output):
        if attn_0_7_output in {"(", "<s>", ")"}:
            return attn_0_0_output == ""

    num_attn_1_0_pattern = select(attn_0_0_outputs, attn_0_7_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 63}:
            return k_mlp_0_0_output == 46
        elif q_mlp_0_0_output in {1, 37, 72, 10, 44, 77, 51}:
            return k_mlp_0_0_output == 33
        elif q_mlp_0_0_output in {2}:
            return k_mlp_0_0_output == 36
        elif q_mlp_0_0_output in {3, 9, 41, 43, 75, 76, 17}:
            return k_mlp_0_0_output == 13
        elif q_mlp_0_0_output in {4, 11, 79, 23, 25}:
            return k_mlp_0_0_output == 19
        elif q_mlp_0_0_output in {16, 5}:
            return k_mlp_0_0_output == 27
        elif q_mlp_0_0_output in {64, 60, 6, 68}:
            return k_mlp_0_0_output == 65
        elif q_mlp_0_0_output in {7}:
            return k_mlp_0_0_output == 30
        elif q_mlp_0_0_output in {8, 13}:
            return k_mlp_0_0_output == 22
        elif q_mlp_0_0_output in {12}:
            return k_mlp_0_0_output == 69
        elif q_mlp_0_0_output in {14}:
            return k_mlp_0_0_output == 75
        elif q_mlp_0_0_output in {15}:
            return k_mlp_0_0_output == 55
        elif q_mlp_0_0_output in {73, 47, 18, 20, 55, 28}:
            return k_mlp_0_0_output == 15
        elif q_mlp_0_0_output in {49, 19}:
            return k_mlp_0_0_output == 24
        elif q_mlp_0_0_output in {45, 21, 62}:
            return k_mlp_0_0_output == 31
        elif q_mlp_0_0_output in {56, 22}:
            return k_mlp_0_0_output == 74
        elif q_mlp_0_0_output in {24}:
            return k_mlp_0_0_output == 16
        elif q_mlp_0_0_output in {48, 26, 30}:
            return k_mlp_0_0_output == 7
        elif q_mlp_0_0_output in {58, 27}:
            return k_mlp_0_0_output == 28
        elif q_mlp_0_0_output in {69, 40, 74, 78, 54, 57, 29, 31}:
            return k_mlp_0_0_output == 26
        elif q_mlp_0_0_output in {32, 71}:
            return k_mlp_0_0_output == 61
        elif q_mlp_0_0_output in {33, 59, 46}:
            return k_mlp_0_0_output == 32
        elif q_mlp_0_0_output in {34}:
            return k_mlp_0_0_output == 20
        elif q_mlp_0_0_output in {35}:
            return k_mlp_0_0_output == 12
        elif q_mlp_0_0_output in {36}:
            return k_mlp_0_0_output == 9
        elif q_mlp_0_0_output in {38}:
            return k_mlp_0_0_output == 6
        elif q_mlp_0_0_output in {39}:
            return k_mlp_0_0_output == 70
        elif q_mlp_0_0_output in {42}:
            return k_mlp_0_0_output == 76
        elif q_mlp_0_0_output in {50}:
            return k_mlp_0_0_output == 23
        elif q_mlp_0_0_output in {52}:
            return k_mlp_0_0_output == 10
        elif q_mlp_0_0_output in {66, 53}:
            return k_mlp_0_0_output == 14
        elif q_mlp_0_0_output in {61}:
            return k_mlp_0_0_output == 43
        elif q_mlp_0_0_output in {65}:
            return k_mlp_0_0_output == 63
        elif q_mlp_0_0_output in {67}:
            return k_mlp_0_0_output == 62
        elif q_mlp_0_0_output in {70}:
            return k_mlp_0_0_output == 49

    num_attn_1_1_pattern = select(mlp_0_0_outputs, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_5_output, mlp_0_0_output):
        if attn_0_5_output in {"(", "<s>", ")"}:
            return mlp_0_0_output == 39

    num_attn_1_2_pattern = select(mlp_0_0_outputs, attn_0_5_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_token, k_token):
        if q_token in {"(", "<s>", ")"}:
            return k_token == ""

    num_attn_1_3_pattern = select(tokens, tokens, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_4_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_6_output, token):
        if attn_0_6_output in {"(", "<s>", ")"}:
            return token == ""

    num_attn_1_4_pattern = select(tokens, attn_0_6_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_4_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_4_output, num_mlp_0_0_output):
        if attn_0_4_output in {"("}:
            return num_mlp_0_0_output == 41
        elif attn_0_4_output in {")"}:
            return num_mlp_0_0_output == 70
        elif attn_0_4_output in {"<s>"}:
            return num_mlp_0_0_output == 18

    num_attn_1_5_pattern = select(
        num_mlp_0_0_outputs, attn_0_4_outputs, num_predicate_1_5
    )
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_5_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(mlp_0_1_output, token):
        if mlp_0_1_output in {
            0,
            1,
            2,
            7,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            19,
            20,
            24,
            25,
            26,
            27,
            31,
            32,
            33,
            34,
            35,
            36,
            38,
            39,
            40,
            42,
            43,
            44,
            46,
            47,
            50,
            51,
            52,
            54,
            56,
            57,
            61,
            63,
            64,
            65,
            67,
            69,
            71,
            73,
            75,
            76,
            78,
        }:
            return token == ""
        elif mlp_0_1_output in {
            3,
            4,
            5,
            6,
            8,
            10,
            17,
            18,
            21,
            22,
            28,
            29,
            30,
            37,
            41,
            45,
            48,
            49,
            53,
            55,
            58,
            59,
            60,
            62,
            66,
            68,
            70,
            72,
            74,
            77,
            79,
        }:
            return token == "("
        elif mlp_0_1_output in {23}:
            return token == "<pad>"

    num_attn_1_6_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_1_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_6_output, mlp_0_0_output):
        if attn_0_6_output in {"("}:
            return mlp_0_0_output == 32
        elif attn_0_6_output in {")"}:
            return mlp_0_0_output == 19
        elif attn_0_6_output in {"<s>"}:
            return mlp_0_0_output == 26

    num_attn_1_7_pattern = select(mlp_0_0_outputs, attn_0_6_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_2_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_7_output, attn_0_5_output):
        key = (attn_0_7_output, attn_0_5_output)
        return 64

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_1_output, attn_1_4_output):
        key = (attn_1_1_output, attn_1_4_output)
        if key in {("(", "(")}:
            return 36
        return 58

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_4_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_4_output, num_attn_1_7_output):
        key = (num_attn_1_4_output, num_attn_1_7_output)
        return 57

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output, num_attn_1_4_output):
        key = (num_attn_0_1_output, num_attn_1_4_output)
        if key in {
            (12, 0),
            (13, 0),
            (14, 0),
            (15, 0),
            (15, 1),
            (16, 0),
            (16, 1),
            (17, 0),
            (17, 1),
            (18, 0),
            (18, 1),
            (19, 0),
            (19, 1),
            (19, 2),
            (20, 0),
            (20, 1),
            (20, 2),
            (21, 0),
            (21, 1),
            (21, 2),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 8),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (43, 9),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (44, 9),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (45, 9),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (46, 9),
            (46, 10),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (47, 6),
            (47, 7),
            (47, 8),
            (47, 9),
            (47, 10),
            (48, 0),
            (48, 1),
            (48, 2),
            (48, 3),
            (48, 4),
            (48, 5),
            (48, 6),
            (48, 7),
            (48, 8),
            (48, 9),
            (48, 10),
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (49, 5),
            (49, 6),
            (49, 7),
            (49, 8),
            (49, 9),
            (49, 10),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (50, 5),
            (50, 6),
            (50, 7),
            (50, 8),
            (50, 9),
            (50, 10),
            (50, 11),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (51, 5),
            (51, 6),
            (51, 7),
            (51, 8),
            (51, 9),
            (51, 10),
            (51, 11),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
            (52, 5),
            (52, 6),
            (52, 7),
            (52, 8),
            (52, 9),
            (52, 10),
            (52, 11),
            (53, 0),
            (53, 1),
            (53, 2),
            (53, 3),
            (53, 4),
            (53, 5),
            (53, 6),
            (53, 7),
            (53, 8),
            (53, 9),
            (53, 10),
            (53, 11),
            (53, 12),
            (54, 0),
            (54, 1),
            (54, 2),
            (54, 3),
            (54, 4),
            (54, 5),
            (54, 6),
            (54, 7),
            (54, 8),
            (54, 9),
            (54, 10),
            (54, 11),
            (54, 12),
            (55, 0),
            (55, 1),
            (55, 2),
            (55, 3),
            (55, 4),
            (55, 5),
            (55, 6),
            (55, 7),
            (55, 8),
            (55, 9),
            (55, 10),
            (55, 11),
            (55, 12),
            (56, 0),
            (56, 1),
            (56, 2),
            (56, 3),
            (56, 4),
            (56, 5),
            (56, 6),
            (56, 7),
            (56, 8),
            (56, 9),
            (56, 10),
            (56, 11),
            (56, 12),
            (57, 0),
            (57, 1),
            (57, 2),
            (57, 3),
            (57, 4),
            (57, 5),
            (57, 6),
            (57, 7),
            (57, 8),
            (57, 9),
            (57, 10),
            (57, 11),
            (57, 12),
            (57, 13),
            (58, 0),
            (58, 1),
            (58, 2),
            (58, 3),
            (58, 4),
            (58, 5),
            (58, 6),
            (58, 7),
            (58, 8),
            (58, 9),
            (58, 10),
            (58, 11),
            (58, 12),
            (58, 13),
            (59, 0),
            (59, 1),
            (59, 2),
            (59, 3),
            (59, 4),
            (59, 5),
            (59, 6),
            (59, 7),
            (59, 8),
            (59, 9),
            (59, 10),
            (59, 11),
            (59, 12),
            (59, 13),
            (60, 0),
            (60, 1),
            (60, 2),
            (60, 3),
            (60, 4),
            (60, 5),
            (60, 6),
            (60, 7),
            (60, 8),
            (60, 9),
            (60, 10),
            (60, 11),
            (60, 12),
            (60, 13),
            (60, 14),
            (61, 0),
            (61, 1),
            (61, 2),
            (61, 3),
            (61, 4),
            (61, 5),
            (61, 6),
            (61, 7),
            (61, 8),
            (61, 9),
            (61, 10),
            (61, 11),
            (61, 12),
            (61, 13),
            (61, 14),
            (62, 0),
            (62, 1),
            (62, 2),
            (62, 3),
            (62, 4),
            (62, 5),
            (62, 6),
            (62, 7),
            (62, 8),
            (62, 9),
            (62, 10),
            (62, 11),
            (62, 12),
            (62, 13),
            (62, 14),
            (63, 0),
            (63, 1),
            (63, 2),
            (63, 3),
            (63, 4),
            (63, 5),
            (63, 6),
            (63, 7),
            (63, 8),
            (63, 9),
            (63, 10),
            (63, 11),
            (63, 12),
            (63, 13),
            (63, 14),
            (64, 0),
            (64, 1),
            (64, 2),
            (64, 3),
            (64, 4),
            (64, 5),
            (64, 6),
            (64, 7),
            (64, 8),
            (64, 9),
            (64, 10),
            (64, 11),
            (64, 12),
            (64, 13),
            (64, 14),
            (64, 15),
            (65, 0),
            (65, 1),
            (65, 2),
            (65, 3),
            (65, 4),
            (65, 5),
            (65, 6),
            (65, 7),
            (65, 8),
            (65, 9),
            (65, 10),
            (65, 11),
            (65, 12),
            (65, 13),
            (65, 14),
            (65, 15),
            (66, 0),
            (66, 1),
            (66, 2),
            (66, 3),
            (66, 4),
            (66, 5),
            (66, 6),
            (66, 7),
            (66, 8),
            (66, 9),
            (66, 10),
            (66, 11),
            (66, 12),
            (66, 13),
            (66, 14),
            (66, 15),
            (67, 0),
            (67, 1),
            (67, 2),
            (67, 3),
            (67, 4),
            (67, 5),
            (67, 6),
            (67, 7),
            (67, 8),
            (67, 9),
            (67, 10),
            (67, 11),
            (67, 12),
            (67, 13),
            (67, 14),
            (67, 15),
            (67, 16),
            (68, 0),
            (68, 1),
            (68, 2),
            (68, 3),
            (68, 4),
            (68, 5),
            (68, 6),
            (68, 7),
            (68, 8),
            (68, 9),
            (68, 10),
            (68, 11),
            (68, 12),
            (68, 13),
            (68, 14),
            (68, 15),
            (68, 16),
            (69, 0),
            (69, 1),
            (69, 2),
            (69, 3),
            (69, 4),
            (69, 5),
            (69, 6),
            (69, 7),
            (69, 8),
            (69, 9),
            (69, 10),
            (69, 11),
            (69, 12),
            (69, 13),
            (69, 14),
            (69, 15),
            (69, 16),
            (70, 0),
            (70, 1),
            (70, 2),
            (70, 3),
            (70, 4),
            (70, 5),
            (70, 6),
            (70, 7),
            (70, 8),
            (70, 9),
            (70, 10),
            (70, 11),
            (70, 12),
            (70, 13),
            (70, 14),
            (70, 15),
            (70, 16),
            (71, 0),
            (71, 1),
            (71, 2),
            (71, 3),
            (71, 4),
            (71, 5),
            (71, 6),
            (71, 7),
            (71, 8),
            (71, 9),
            (71, 10),
            (71, 11),
            (71, 12),
            (71, 13),
            (71, 14),
            (71, 15),
            (71, 16),
            (71, 17),
            (72, 0),
            (72, 1),
            (72, 2),
            (72, 3),
            (72, 4),
            (72, 5),
            (72, 6),
            (72, 7),
            (72, 8),
            (72, 9),
            (72, 10),
            (72, 11),
            (72, 12),
            (72, 13),
            (72, 14),
            (72, 15),
            (72, 16),
            (72, 17),
            (73, 0),
            (73, 1),
            (73, 2),
            (73, 3),
            (73, 4),
            (73, 5),
            (73, 6),
            (73, 7),
            (73, 8),
            (73, 9),
            (73, 10),
            (73, 11),
            (73, 12),
            (73, 13),
            (73, 14),
            (73, 15),
            (73, 16),
            (73, 17),
            (74, 0),
            (74, 1),
            (74, 2),
            (74, 3),
            (74, 4),
            (74, 5),
            (74, 6),
            (74, 7),
            (74, 8),
            (74, 9),
            (74, 10),
            (74, 11),
            (74, 12),
            (74, 13),
            (74, 14),
            (74, 15),
            (74, 16),
            (74, 17),
            (74, 18),
            (75, 0),
            (75, 1),
            (75, 2),
            (75, 3),
            (75, 4),
            (75, 5),
            (75, 6),
            (75, 7),
            (75, 8),
            (75, 9),
            (75, 10),
            (75, 11),
            (75, 12),
            (75, 13),
            (75, 14),
            (75, 15),
            (75, 16),
            (75, 17),
            (75, 18),
            (76, 0),
            (76, 1),
            (76, 2),
            (76, 3),
            (76, 4),
            (76, 5),
            (76, 6),
            (76, 7),
            (76, 8),
            (76, 9),
            (76, 10),
            (76, 11),
            (76, 12),
            (76, 13),
            (76, 14),
            (76, 15),
            (76, 16),
            (76, 17),
            (76, 18),
            (77, 0),
            (77, 1),
            (77, 2),
            (77, 3),
            (77, 4),
            (77, 5),
            (77, 6),
            (77, 7),
            (77, 8),
            (77, 9),
            (77, 10),
            (77, 11),
            (77, 12),
            (77, 13),
            (77, 14),
            (77, 15),
            (77, 16),
            (77, 17),
            (77, 18),
            (78, 0),
            (78, 1),
            (78, 2),
            (78, 3),
            (78, 4),
            (78, 5),
            (78, 6),
            (78, 7),
            (78, 8),
            (78, 9),
            (78, 10),
            (78, 11),
            (78, 12),
            (78, 13),
            (78, 14),
            (78, 15),
            (78, 16),
            (78, 17),
            (78, 18),
            (78, 19),
            (79, 0),
            (79, 1),
            (79, 2),
            (79, 3),
            (79, 4),
            (79, 5),
            (79, 6),
            (79, 7),
            (79, 8),
            (79, 9),
            (79, 10),
            (79, 11),
            (79, 12),
            (79, 13),
            (79, 14),
            (79, 15),
            (79, 16),
            (79, 17),
            (79, 18),
            (79, 19),
            (80, 0),
            (80, 1),
            (80, 2),
            (80, 3),
            (80, 4),
            (80, 5),
            (80, 6),
            (80, 7),
            (80, 8),
            (80, 9),
            (80, 10),
            (80, 11),
            (80, 12),
            (80, 13),
            (80, 14),
            (80, 15),
            (80, 16),
            (80, 17),
            (80, 18),
            (80, 19),
            (81, 0),
            (81, 1),
            (81, 2),
            (81, 3),
            (81, 4),
            (81, 5),
            (81, 6),
            (81, 7),
            (81, 8),
            (81, 9),
            (81, 10),
            (81, 11),
            (81, 12),
            (81, 13),
            (81, 14),
            (81, 15),
            (81, 16),
            (81, 17),
            (81, 18),
            (81, 19),
            (81, 20),
            (82, 0),
            (82, 1),
            (82, 2),
            (82, 3),
            (82, 4),
            (82, 5),
            (82, 6),
            (82, 7),
            (82, 8),
            (82, 9),
            (82, 10),
            (82, 11),
            (82, 12),
            (82, 13),
            (82, 14),
            (82, 15),
            (82, 16),
            (82, 17),
            (82, 18),
            (82, 19),
            (82, 20),
            (83, 0),
            (83, 1),
            (83, 2),
            (83, 3),
            (83, 4),
            (83, 5),
            (83, 6),
            (83, 7),
            (83, 8),
            (83, 9),
            (83, 10),
            (83, 11),
            (83, 12),
            (83, 13),
            (83, 14),
            (83, 15),
            (83, 16),
            (83, 17),
            (83, 18),
            (83, 19),
            (83, 20),
            (84, 0),
            (84, 1),
            (84, 2),
            (84, 3),
            (84, 4),
            (84, 5),
            (84, 6),
            (84, 7),
            (84, 8),
            (84, 9),
            (84, 10),
            (84, 11),
            (84, 12),
            (84, 13),
            (84, 14),
            (84, 15),
            (84, 16),
            (84, 17),
            (84, 18),
            (84, 19),
            (84, 20),
            (85, 0),
            (85, 1),
            (85, 2),
            (85, 3),
            (85, 4),
            (85, 5),
            (85, 6),
            (85, 7),
            (85, 8),
            (85, 9),
            (85, 10),
            (85, 11),
            (85, 12),
            (85, 13),
            (85, 14),
            (85, 15),
            (85, 16),
            (85, 17),
            (85, 18),
            (85, 19),
            (85, 20),
            (85, 21),
            (86, 0),
            (86, 1),
            (86, 2),
            (86, 3),
            (86, 4),
            (86, 5),
            (86, 6),
            (86, 7),
            (86, 8),
            (86, 9),
            (86, 10),
            (86, 11),
            (86, 12),
            (86, 13),
            (86, 14),
            (86, 15),
            (86, 16),
            (86, 17),
            (86, 18),
            (86, 19),
            (86, 20),
            (86, 21),
            (87, 0),
            (87, 1),
            (87, 2),
            (87, 3),
            (87, 4),
            (87, 5),
            (87, 6),
            (87, 7),
            (87, 8),
            (87, 9),
            (87, 10),
            (87, 11),
            (87, 12),
            (87, 13),
            (87, 14),
            (87, 15),
            (87, 16),
            (87, 17),
            (87, 18),
            (87, 19),
            (87, 20),
            (87, 21),
            (88, 0),
            (88, 1),
            (88, 2),
            (88, 3),
            (88, 4),
            (88, 5),
            (88, 6),
            (88, 7),
            (88, 8),
            (88, 9),
            (88, 10),
            (88, 11),
            (88, 12),
            (88, 13),
            (88, 14),
            (88, 15),
            (88, 16),
            (88, 17),
            (88, 18),
            (88, 19),
            (88, 20),
            (88, 21),
            (88, 22),
            (89, 0),
            (89, 1),
            (89, 2),
            (89, 3),
            (89, 4),
            (89, 5),
            (89, 6),
            (89, 7),
            (89, 8),
            (89, 9),
            (89, 10),
            (89, 11),
            (89, 12),
            (89, 13),
            (89, 14),
            (89, 15),
            (89, 16),
            (89, 17),
            (89, 18),
            (89, 19),
            (89, 20),
            (89, 21),
            (89, 22),
            (90, 0),
            (90, 1),
            (90, 2),
            (90, 3),
            (90, 4),
            (90, 5),
            (90, 6),
            (90, 7),
            (90, 8),
            (90, 9),
            (90, 10),
            (90, 11),
            (90, 12),
            (90, 13),
            (90, 14),
            (90, 15),
            (90, 16),
            (90, 17),
            (90, 18),
            (90, 19),
            (90, 20),
            (90, 21),
            (90, 22),
            (91, 0),
            (91, 1),
            (91, 2),
            (91, 3),
            (91, 4),
            (91, 5),
            (91, 6),
            (91, 7),
            (91, 8),
            (91, 9),
            (91, 10),
            (91, 11),
            (91, 12),
            (91, 13),
            (91, 14),
            (91, 15),
            (91, 16),
            (91, 17),
            (91, 18),
            (91, 19),
            (91, 20),
            (91, 21),
            (91, 22),
            (92, 0),
            (92, 1),
            (92, 2),
            (92, 3),
            (92, 4),
            (92, 5),
            (92, 6),
            (92, 7),
            (92, 8),
            (92, 9),
            (92, 10),
            (92, 11),
            (92, 12),
            (92, 13),
            (92, 14),
            (92, 15),
            (92, 16),
            (92, 17),
            (92, 18),
            (92, 19),
            (92, 20),
            (92, 21),
            (92, 22),
            (92, 23),
            (93, 0),
            (93, 1),
            (93, 2),
            (93, 3),
            (93, 4),
            (93, 5),
            (93, 6),
            (93, 7),
            (93, 8),
            (93, 9),
            (93, 10),
            (93, 11),
            (93, 12),
            (93, 13),
            (93, 14),
            (93, 15),
            (93, 16),
            (93, 17),
            (93, 18),
            (93, 19),
            (93, 20),
            (93, 21),
            (93, 22),
            (93, 23),
            (94, 0),
            (94, 1),
            (94, 2),
            (94, 3),
            (94, 4),
            (94, 5),
            (94, 6),
            (94, 7),
            (94, 8),
            (94, 9),
            (94, 10),
            (94, 11),
            (94, 12),
            (94, 13),
            (94, 14),
            (94, 15),
            (94, 16),
            (94, 17),
            (94, 18),
            (94, 19),
            (94, 20),
            (94, 21),
            (94, 22),
            (94, 23),
            (95, 0),
            (95, 1),
            (95, 2),
            (95, 3),
            (95, 4),
            (95, 5),
            (95, 6),
            (95, 7),
            (95, 8),
            (95, 9),
            (95, 10),
            (95, 11),
            (95, 12),
            (95, 13),
            (95, 14),
            (95, 15),
            (95, 16),
            (95, 17),
            (95, 18),
            (95, 19),
            (95, 20),
            (95, 21),
            (95, 22),
            (95, 23),
            (95, 24),
            (96, 0),
            (96, 1),
            (96, 2),
            (96, 3),
            (96, 4),
            (96, 5),
            (96, 6),
            (96, 7),
            (96, 8),
            (96, 9),
            (96, 10),
            (96, 11),
            (96, 12),
            (96, 13),
            (96, 14),
            (96, 15),
            (96, 16),
            (96, 17),
            (96, 18),
            (96, 19),
            (96, 20),
            (96, 21),
            (96, 22),
            (96, 23),
            (96, 24),
            (97, 0),
            (97, 1),
            (97, 2),
            (97, 3),
            (97, 4),
            (97, 5),
            (97, 6),
            (97, 7),
            (97, 8),
            (97, 9),
            (97, 10),
            (97, 11),
            (97, 12),
            (97, 13),
            (97, 14),
            (97, 15),
            (97, 16),
            (97, 17),
            (97, 18),
            (97, 19),
            (97, 20),
            (97, 21),
            (97, 22),
            (97, 23),
            (97, 24),
            (98, 0),
            (98, 1),
            (98, 2),
            (98, 3),
            (98, 4),
            (98, 5),
            (98, 6),
            (98, 7),
            (98, 8),
            (98, 9),
            (98, 10),
            (98, 11),
            (98, 12),
            (98, 13),
            (98, 14),
            (98, 15),
            (98, 16),
            (98, 17),
            (98, 18),
            (98, 19),
            (98, 20),
            (98, 21),
            (98, 22),
            (98, 23),
            (98, 24),
            (99, 0),
            (99, 1),
            (99, 2),
            (99, 3),
            (99, 4),
            (99, 5),
            (99, 6),
            (99, 7),
            (99, 8),
            (99, 9),
            (99, 10),
            (99, 11),
            (99, 12),
            (99, 13),
            (99, 14),
            (99, 15),
            (99, 16),
            (99, 17),
            (99, 18),
            (99, 19),
            (99, 20),
            (99, 21),
            (99, 22),
            (99, 23),
            (99, 24),
            (99, 25),
            (100, 0),
            (100, 1),
            (100, 2),
            (100, 3),
            (100, 4),
            (100, 5),
            (100, 6),
            (100, 7),
            (100, 8),
            (100, 9),
            (100, 10),
            (100, 11),
            (100, 12),
            (100, 13),
            (100, 14),
            (100, 15),
            (100, 16),
            (100, 17),
            (100, 18),
            (100, 19),
            (100, 20),
            (100, 21),
            (100, 22),
            (100, 23),
            (100, 24),
            (100, 25),
            (101, 0),
            (101, 1),
            (101, 2),
            (101, 3),
            (101, 4),
            (101, 5),
            (101, 6),
            (101, 7),
            (101, 8),
            (101, 9),
            (101, 10),
            (101, 11),
            (101, 12),
            (101, 13),
            (101, 14),
            (101, 15),
            (101, 16),
            (101, 17),
            (101, 18),
            (101, 19),
            (101, 20),
            (101, 21),
            (101, 22),
            (101, 23),
            (101, 24),
            (101, 25),
            (102, 0),
            (102, 1),
            (102, 2),
            (102, 3),
            (102, 4),
            (102, 5),
            (102, 6),
            (102, 7),
            (102, 8),
            (102, 9),
            (102, 10),
            (102, 11),
            (102, 12),
            (102, 13),
            (102, 14),
            (102, 15),
            (102, 16),
            (102, 17),
            (102, 18),
            (102, 19),
            (102, 20),
            (102, 21),
            (102, 22),
            (102, 23),
            (102, 24),
            (102, 25),
            (102, 26),
            (103, 0),
            (103, 1),
            (103, 2),
            (103, 3),
            (103, 4),
            (103, 5),
            (103, 6),
            (103, 7),
            (103, 8),
            (103, 9),
            (103, 10),
            (103, 11),
            (103, 12),
            (103, 13),
            (103, 14),
            (103, 15),
            (103, 16),
            (103, 17),
            (103, 18),
            (103, 19),
            (103, 20),
            (103, 21),
            (103, 22),
            (103, 23),
            (103, 24),
            (103, 25),
            (103, 26),
            (104, 0),
            (104, 1),
            (104, 2),
            (104, 3),
            (104, 4),
            (104, 5),
            (104, 6),
            (104, 7),
            (104, 8),
            (104, 9),
            (104, 10),
            (104, 11),
            (104, 12),
            (104, 13),
            (104, 14),
            (104, 15),
            (104, 16),
            (104, 17),
            (104, 18),
            (104, 19),
            (104, 20),
            (104, 21),
            (104, 22),
            (104, 23),
            (104, 24),
            (104, 25),
            (104, 26),
            (105, 0),
            (105, 1),
            (105, 2),
            (105, 3),
            (105, 4),
            (105, 5),
            (105, 6),
            (105, 7),
            (105, 8),
            (105, 9),
            (105, 10),
            (105, 11),
            (105, 12),
            (105, 13),
            (105, 14),
            (105, 15),
            (105, 16),
            (105, 17),
            (105, 18),
            (105, 19),
            (105, 20),
            (105, 21),
            (105, 22),
            (105, 23),
            (105, 24),
            (105, 25),
            (105, 26),
            (106, 0),
            (106, 1),
            (106, 2),
            (106, 3),
            (106, 4),
            (106, 5),
            (106, 6),
            (106, 7),
            (106, 8),
            (106, 9),
            (106, 10),
            (106, 11),
            (106, 12),
            (106, 13),
            (106, 14),
            (106, 15),
            (106, 16),
            (106, 17),
            (106, 18),
            (106, 19),
            (106, 20),
            (106, 21),
            (106, 22),
            (106, 23),
            (106, 24),
            (106, 25),
            (106, 26),
            (106, 27),
            (107, 0),
            (107, 1),
            (107, 2),
            (107, 3),
            (107, 4),
            (107, 5),
            (107, 6),
            (107, 7),
            (107, 8),
            (107, 9),
            (107, 10),
            (107, 11),
            (107, 12),
            (107, 13),
            (107, 14),
            (107, 15),
            (107, 16),
            (107, 17),
            (107, 18),
            (107, 19),
            (107, 20),
            (107, 21),
            (107, 22),
            (107, 23),
            (107, 24),
            (107, 25),
            (107, 26),
            (107, 27),
            (108, 0),
            (108, 1),
            (108, 2),
            (108, 3),
            (108, 4),
            (108, 5),
            (108, 6),
            (108, 7),
            (108, 8),
            (108, 9),
            (108, 10),
            (108, 11),
            (108, 12),
            (108, 13),
            (108, 14),
            (108, 15),
            (108, 16),
            (108, 17),
            (108, 18),
            (108, 19),
            (108, 20),
            (108, 21),
            (108, 22),
            (108, 23),
            (108, 24),
            (108, 25),
            (108, 26),
            (108, 27),
            (109, 0),
            (109, 1),
            (109, 2),
            (109, 3),
            (109, 4),
            (109, 5),
            (109, 6),
            (109, 7),
            (109, 8),
            (109, 9),
            (109, 10),
            (109, 11),
            (109, 12),
            (109, 13),
            (109, 14),
            (109, 15),
            (109, 16),
            (109, 17),
            (109, 18),
            (109, 19),
            (109, 20),
            (109, 21),
            (109, 22),
            (109, 23),
            (109, 24),
            (109, 25),
            (109, 26),
            (109, 27),
            (109, 28),
            (110, 0),
            (110, 1),
            (110, 2),
            (110, 3),
            (110, 4),
            (110, 5),
            (110, 6),
            (110, 7),
            (110, 8),
            (110, 9),
            (110, 10),
            (110, 11),
            (110, 12),
            (110, 13),
            (110, 14),
            (110, 15),
            (110, 16),
            (110, 17),
            (110, 18),
            (110, 19),
            (110, 20),
            (110, 21),
            (110, 22),
            (110, 23),
            (110, 24),
            (110, 25),
            (110, 26),
            (110, 27),
            (110, 28),
            (111, 0),
            (111, 1),
            (111, 2),
            (111, 3),
            (111, 4),
            (111, 5),
            (111, 6),
            (111, 7),
            (111, 8),
            (111, 9),
            (111, 10),
            (111, 11),
            (111, 12),
            (111, 13),
            (111, 14),
            (111, 15),
            (111, 16),
            (111, 17),
            (111, 18),
            (111, 19),
            (111, 20),
            (111, 21),
            (111, 22),
            (111, 23),
            (111, 24),
            (111, 25),
            (111, 26),
            (111, 27),
            (111, 28),
            (112, 0),
            (112, 1),
            (112, 2),
            (112, 3),
            (112, 4),
            (112, 5),
            (112, 6),
            (112, 7),
            (112, 8),
            (112, 9),
            (112, 10),
            (112, 11),
            (112, 12),
            (112, 13),
            (112, 14),
            (112, 15),
            (112, 16),
            (112, 17),
            (112, 18),
            (112, 19),
            (112, 20),
            (112, 21),
            (112, 22),
            (112, 23),
            (112, 24),
            (112, 25),
            (112, 26),
            (112, 27),
            (112, 28),
            (113, 0),
            (113, 1),
            (113, 2),
            (113, 3),
            (113, 4),
            (113, 5),
            (113, 6),
            (113, 7),
            (113, 8),
            (113, 9),
            (113, 10),
            (113, 11),
            (113, 12),
            (113, 13),
            (113, 14),
            (113, 15),
            (113, 16),
            (113, 17),
            (113, 18),
            (113, 19),
            (113, 20),
            (113, 21),
            (113, 22),
            (113, 23),
            (113, 24),
            (113, 25),
            (113, 26),
            (113, 27),
            (113, 28),
            (113, 29),
            (114, 0),
            (114, 1),
            (114, 2),
            (114, 3),
            (114, 4),
            (114, 5),
            (114, 6),
            (114, 7),
            (114, 8),
            (114, 9),
            (114, 10),
            (114, 11),
            (114, 12),
            (114, 13),
            (114, 14),
            (114, 15),
            (114, 16),
            (114, 17),
            (114, 18),
            (114, 19),
            (114, 20),
            (114, 21),
            (114, 22),
            (114, 23),
            (114, 24),
            (114, 25),
            (114, 26),
            (114, 27),
            (114, 28),
            (114, 29),
            (115, 0),
            (115, 1),
            (115, 2),
            (115, 3),
            (115, 4),
            (115, 5),
            (115, 6),
            (115, 7),
            (115, 8),
            (115, 9),
            (115, 10),
            (115, 11),
            (115, 12),
            (115, 13),
            (115, 14),
            (115, 15),
            (115, 16),
            (115, 17),
            (115, 18),
            (115, 19),
            (115, 20),
            (115, 21),
            (115, 22),
            (115, 23),
            (115, 24),
            (115, 25),
            (115, 26),
            (115, 27),
            (115, 28),
            (115, 29),
            (116, 0),
            (116, 1),
            (116, 2),
            (116, 3),
            (116, 4),
            (116, 5),
            (116, 6),
            (116, 7),
            (116, 8),
            (116, 9),
            (116, 10),
            (116, 11),
            (116, 12),
            (116, 13),
            (116, 14),
            (116, 15),
            (116, 16),
            (116, 17),
            (116, 18),
            (116, 19),
            (116, 20),
            (116, 21),
            (116, 22),
            (116, 23),
            (116, 24),
            (116, 25),
            (116, 26),
            (116, 27),
            (116, 28),
            (116, 29),
            (116, 30),
            (117, 0),
            (117, 1),
            (117, 2),
            (117, 3),
            (117, 4),
            (117, 5),
            (117, 6),
            (117, 7),
            (117, 8),
            (117, 9),
            (117, 10),
            (117, 11),
            (117, 12),
            (117, 13),
            (117, 14),
            (117, 15),
            (117, 16),
            (117, 17),
            (117, 18),
            (117, 19),
            (117, 20),
            (117, 21),
            (117, 22),
            (117, 23),
            (117, 24),
            (117, 25),
            (117, 26),
            (117, 27),
            (117, 28),
            (117, 29),
            (117, 30),
            (118, 0),
            (118, 1),
            (118, 2),
            (118, 3),
            (118, 4),
            (118, 5),
            (118, 6),
            (118, 7),
            (118, 8),
            (118, 9),
            (118, 10),
            (118, 11),
            (118, 12),
            (118, 13),
            (118, 14),
            (118, 15),
            (118, 16),
            (118, 17),
            (118, 18),
            (118, 19),
            (118, 20),
            (118, 21),
            (118, 22),
            (118, 23),
            (118, 24),
            (118, 25),
            (118, 26),
            (118, 27),
            (118, 28),
            (118, 29),
            (118, 30),
            (119, 0),
            (119, 1),
            (119, 2),
            (119, 3),
            (119, 4),
            (119, 5),
            (119, 6),
            (119, 7),
            (119, 8),
            (119, 9),
            (119, 10),
            (119, 11),
            (119, 12),
            (119, 13),
            (119, 14),
            (119, 15),
            (119, 16),
            (119, 17),
            (119, 18),
            (119, 19),
            (119, 20),
            (119, 21),
            (119, 22),
            (119, 23),
            (119, 24),
            (119, 25),
            (119, 26),
            (119, 27),
            (119, 28),
            (119, 29),
            (119, 30),
            (120, 0),
            (120, 1),
            (120, 2),
            (120, 3),
            (120, 4),
            (120, 5),
            (120, 6),
            (120, 7),
            (120, 8),
            (120, 9),
            (120, 10),
            (120, 11),
            (120, 12),
            (120, 13),
            (120, 14),
            (120, 15),
            (120, 16),
            (120, 17),
            (120, 18),
            (120, 19),
            (120, 20),
            (120, 21),
            (120, 22),
            (120, 23),
            (120, 24),
            (120, 25),
            (120, 26),
            (120, 27),
            (120, 28),
            (120, 29),
            (120, 30),
            (120, 31),
            (121, 0),
            (121, 1),
            (121, 2),
            (121, 3),
            (121, 4),
            (121, 5),
            (121, 6),
            (121, 7),
            (121, 8),
            (121, 9),
            (121, 10),
            (121, 11),
            (121, 12),
            (121, 13),
            (121, 14),
            (121, 15),
            (121, 16),
            (121, 17),
            (121, 18),
            (121, 19),
            (121, 20),
            (121, 21),
            (121, 22),
            (121, 23),
            (121, 24),
            (121, 25),
            (121, 26),
            (121, 27),
            (121, 28),
            (121, 29),
            (121, 30),
            (121, 31),
            (122, 0),
            (122, 1),
            (122, 2),
            (122, 3),
            (122, 4),
            (122, 5),
            (122, 6),
            (122, 7),
            (122, 8),
            (122, 9),
            (122, 10),
            (122, 11),
            (122, 12),
            (122, 13),
            (122, 14),
            (122, 15),
            (122, 16),
            (122, 17),
            (122, 18),
            (122, 19),
            (122, 20),
            (122, 21),
            (122, 22),
            (122, 23),
            (122, 24),
            (122, 25),
            (122, 26),
            (122, 27),
            (122, 28),
            (122, 29),
            (122, 30),
            (122, 31),
            (123, 0),
            (123, 1),
            (123, 2),
            (123, 3),
            (123, 4),
            (123, 5),
            (123, 6),
            (123, 7),
            (123, 8),
            (123, 9),
            (123, 10),
            (123, 11),
            (123, 12),
            (123, 13),
            (123, 14),
            (123, 15),
            (123, 16),
            (123, 17),
            (123, 18),
            (123, 19),
            (123, 20),
            (123, 21),
            (123, 22),
            (123, 23),
            (123, 24),
            (123, 25),
            (123, 26),
            (123, 27),
            (123, 28),
            (123, 29),
            (123, 30),
            (123, 31),
            (123, 32),
            (124, 0),
            (124, 1),
            (124, 2),
            (124, 3),
            (124, 4),
            (124, 5),
            (124, 6),
            (124, 7),
            (124, 8),
            (124, 9),
            (124, 10),
            (124, 11),
            (124, 12),
            (124, 13),
            (124, 14),
            (124, 15),
            (124, 16),
            (124, 17),
            (124, 18),
            (124, 19),
            (124, 20),
            (124, 21),
            (124, 22),
            (124, 23),
            (124, 24),
            (124, 25),
            (124, 26),
            (124, 27),
            (124, 28),
            (124, 29),
            (124, 30),
            (124, 31),
            (124, 32),
            (125, 0),
            (125, 1),
            (125, 2),
            (125, 3),
            (125, 4),
            (125, 5),
            (125, 6),
            (125, 7),
            (125, 8),
            (125, 9),
            (125, 10),
            (125, 11),
            (125, 12),
            (125, 13),
            (125, 14),
            (125, 15),
            (125, 16),
            (125, 17),
            (125, 18),
            (125, 19),
            (125, 20),
            (125, 21),
            (125, 22),
            (125, 23),
            (125, 24),
            (125, 25),
            (125, 26),
            (125, 27),
            (125, 28),
            (125, 29),
            (125, 30),
            (125, 31),
            (125, 32),
            (126, 0),
            (126, 1),
            (126, 2),
            (126, 3),
            (126, 4),
            (126, 5),
            (126, 6),
            (126, 7),
            (126, 8),
            (126, 9),
            (126, 10),
            (126, 11),
            (126, 12),
            (126, 13),
            (126, 14),
            (126, 15),
            (126, 16),
            (126, 17),
            (126, 18),
            (126, 19),
            (126, 20),
            (126, 21),
            (126, 22),
            (126, 23),
            (126, 24),
            (126, 25),
            (126, 26),
            (126, 27),
            (126, 28),
            (126, 29),
            (126, 30),
            (126, 31),
            (126, 32),
            (127, 0),
            (127, 1),
            (127, 2),
            (127, 3),
            (127, 4),
            (127, 5),
            (127, 6),
            (127, 7),
            (127, 8),
            (127, 9),
            (127, 10),
            (127, 11),
            (127, 12),
            (127, 13),
            (127, 14),
            (127, 15),
            (127, 16),
            (127, 17),
            (127, 18),
            (127, 19),
            (127, 20),
            (127, 21),
            (127, 22),
            (127, 23),
            (127, 24),
            (127, 25),
            (127, 26),
            (127, 27),
            (127, 28),
            (127, 29),
            (127, 30),
            (127, 31),
            (127, 32),
            (127, 33),
            (128, 0),
            (128, 1),
            (128, 2),
            (128, 3),
            (128, 4),
            (128, 5),
            (128, 6),
            (128, 7),
            (128, 8),
            (128, 9),
            (128, 10),
            (128, 11),
            (128, 12),
            (128, 13),
            (128, 14),
            (128, 15),
            (128, 16),
            (128, 17),
            (128, 18),
            (128, 19),
            (128, 20),
            (128, 21),
            (128, 22),
            (128, 23),
            (128, 24),
            (128, 25),
            (128, 26),
            (128, 27),
            (128, 28),
            (128, 29),
            (128, 30),
            (128, 31),
            (128, 32),
            (128, 33),
            (129, 0),
            (129, 1),
            (129, 2),
            (129, 3),
            (129, 4),
            (129, 5),
            (129, 6),
            (129, 7),
            (129, 8),
            (129, 9),
            (129, 10),
            (129, 11),
            (129, 12),
            (129, 13),
            (129, 14),
            (129, 15),
            (129, 16),
            (129, 17),
            (129, 18),
            (129, 19),
            (129, 20),
            (129, 21),
            (129, 22),
            (129, 23),
            (129, 24),
            (129, 25),
            (129, 26),
            (129, 27),
            (129, 28),
            (129, 29),
            (129, 30),
            (129, 31),
            (129, 32),
            (129, 33),
            (130, 0),
            (130, 1),
            (130, 2),
            (130, 3),
            (130, 4),
            (130, 5),
            (130, 6),
            (130, 7),
            (130, 8),
            (130, 9),
            (130, 10),
            (130, 11),
            (130, 12),
            (130, 13),
            (130, 14),
            (130, 15),
            (130, 16),
            (130, 17),
            (130, 18),
            (130, 19),
            (130, 20),
            (130, 21),
            (130, 22),
            (130, 23),
            (130, 24),
            (130, 25),
            (130, 26),
            (130, 27),
            (130, 28),
            (130, 29),
            (130, 30),
            (130, 31),
            (130, 32),
            (130, 33),
            (130, 34),
            (131, 0),
            (131, 1),
            (131, 2),
            (131, 3),
            (131, 4),
            (131, 5),
            (131, 6),
            (131, 7),
            (131, 8),
            (131, 9),
            (131, 10),
            (131, 11),
            (131, 12),
            (131, 13),
            (131, 14),
            (131, 15),
            (131, 16),
            (131, 17),
            (131, 18),
            (131, 19),
            (131, 20),
            (131, 21),
            (131, 22),
            (131, 23),
            (131, 24),
            (131, 25),
            (131, 26),
            (131, 27),
            (131, 28),
            (131, 29),
            (131, 30),
            (131, 31),
            (131, 32),
            (131, 33),
            (131, 34),
            (132, 0),
            (132, 1),
            (132, 2),
            (132, 3),
            (132, 4),
            (132, 5),
            (132, 6),
            (132, 7),
            (132, 8),
            (132, 9),
            (132, 10),
            (132, 11),
            (132, 12),
            (132, 13),
            (132, 14),
            (132, 15),
            (132, 16),
            (132, 17),
            (132, 18),
            (132, 19),
            (132, 20),
            (132, 21),
            (132, 22),
            (132, 23),
            (132, 24),
            (132, 25),
            (132, 26),
            (132, 27),
            (132, 28),
            (132, 29),
            (132, 30),
            (132, 31),
            (132, 32),
            (132, 33),
            (132, 34),
            (133, 0),
            (133, 1),
            (133, 2),
            (133, 3),
            (133, 4),
            (133, 5),
            (133, 6),
            (133, 7),
            (133, 8),
            (133, 9),
            (133, 10),
            (133, 11),
            (133, 12),
            (133, 13),
            (133, 14),
            (133, 15),
            (133, 16),
            (133, 17),
            (133, 18),
            (133, 19),
            (133, 20),
            (133, 21),
            (133, 22),
            (133, 23),
            (133, 24),
            (133, 25),
            (133, 26),
            (133, 27),
            (133, 28),
            (133, 29),
            (133, 30),
            (133, 31),
            (133, 32),
            (133, 33),
            (133, 34),
            (134, 0),
            (134, 1),
            (134, 2),
            (134, 3),
            (134, 4),
            (134, 5),
            (134, 6),
            (134, 7),
            (134, 8),
            (134, 9),
            (134, 10),
            (134, 11),
            (134, 12),
            (134, 13),
            (134, 14),
            (134, 15),
            (134, 16),
            (134, 17),
            (134, 18),
            (134, 19),
            (134, 20),
            (134, 21),
            (134, 22),
            (134, 23),
            (134, 24),
            (134, 25),
            (134, 26),
            (134, 27),
            (134, 28),
            (134, 29),
            (134, 30),
            (134, 31),
            (134, 32),
            (134, 33),
            (134, 34),
            (134, 35),
            (135, 0),
            (135, 1),
            (135, 2),
            (135, 3),
            (135, 4),
            (135, 5),
            (135, 6),
            (135, 7),
            (135, 8),
            (135, 9),
            (135, 10),
            (135, 11),
            (135, 12),
            (135, 13),
            (135, 14),
            (135, 15),
            (135, 16),
            (135, 17),
            (135, 18),
            (135, 19),
            (135, 20),
            (135, 21),
            (135, 22),
            (135, 23),
            (135, 24),
            (135, 25),
            (135, 26),
            (135, 27),
            (135, 28),
            (135, 29),
            (135, 30),
            (135, 31),
            (135, 32),
            (135, 33),
            (135, 34),
            (135, 35),
            (136, 0),
            (136, 1),
            (136, 2),
            (136, 3),
            (136, 4),
            (136, 5),
            (136, 6),
            (136, 7),
            (136, 8),
            (136, 9),
            (136, 10),
            (136, 11),
            (136, 12),
            (136, 13),
            (136, 14),
            (136, 15),
            (136, 16),
            (136, 17),
            (136, 18),
            (136, 19),
            (136, 20),
            (136, 21),
            (136, 22),
            (136, 23),
            (136, 24),
            (136, 25),
            (136, 26),
            (136, 27),
            (136, 28),
            (136, 29),
            (136, 30),
            (136, 31),
            (136, 32),
            (136, 33),
            (136, 34),
            (136, 35),
            (137, 0),
            (137, 1),
            (137, 2),
            (137, 3),
            (137, 4),
            (137, 5),
            (137, 6),
            (137, 7),
            (137, 8),
            (137, 9),
            (137, 10),
            (137, 11),
            (137, 12),
            (137, 13),
            (137, 14),
            (137, 15),
            (137, 16),
            (137, 17),
            (137, 18),
            (137, 19),
            (137, 20),
            (137, 21),
            (137, 22),
            (137, 23),
            (137, 24),
            (137, 25),
            (137, 26),
            (137, 27),
            (137, 28),
            (137, 29),
            (137, 30),
            (137, 31),
            (137, 32),
            (137, 33),
            (137, 34),
            (137, 35),
            (137, 36),
            (138, 0),
            (138, 1),
            (138, 2),
            (138, 3),
            (138, 4),
            (138, 5),
            (138, 6),
            (138, 7),
            (138, 8),
            (138, 9),
            (138, 10),
            (138, 11),
            (138, 12),
            (138, 13),
            (138, 14),
            (138, 15),
            (138, 16),
            (138, 17),
            (138, 18),
            (138, 19),
            (138, 20),
            (138, 21),
            (138, 22),
            (138, 23),
            (138, 24),
            (138, 25),
            (138, 26),
            (138, 27),
            (138, 28),
            (138, 29),
            (138, 30),
            (138, 31),
            (138, 32),
            (138, 33),
            (138, 34),
            (138, 35),
            (138, 36),
            (139, 0),
            (139, 1),
            (139, 2),
            (139, 3),
            (139, 4),
            (139, 5),
            (139, 6),
            (139, 7),
            (139, 8),
            (139, 9),
            (139, 10),
            (139, 11),
            (139, 12),
            (139, 13),
            (139, 14),
            (139, 15),
            (139, 16),
            (139, 17),
            (139, 18),
            (139, 19),
            (139, 20),
            (139, 21),
            (139, 22),
            (139, 23),
            (139, 24),
            (139, 25),
            (139, 26),
            (139, 27),
            (139, 28),
            (139, 29),
            (139, 30),
            (139, 31),
            (139, 32),
            (139, 33),
            (139, 34),
            (139, 35),
            (139, 36),
            (140, 0),
            (140, 1),
            (140, 2),
            (140, 3),
            (140, 4),
            (140, 5),
            (140, 6),
            (140, 7),
            (140, 8),
            (140, 9),
            (140, 10),
            (140, 11),
            (140, 12),
            (140, 13),
            (140, 14),
            (140, 15),
            (140, 16),
            (140, 17),
            (140, 18),
            (140, 19),
            (140, 20),
            (140, 21),
            (140, 22),
            (140, 23),
            (140, 24),
            (140, 25),
            (140, 26),
            (140, 27),
            (140, 28),
            (140, 29),
            (140, 30),
            (140, 31),
            (140, 32),
            (140, 33),
            (140, 34),
            (140, 35),
            (140, 36),
            (141, 0),
            (141, 1),
            (141, 2),
            (141, 3),
            (141, 4),
            (141, 5),
            (141, 6),
            (141, 7),
            (141, 8),
            (141, 9),
            (141, 10),
            (141, 11),
            (141, 12),
            (141, 13),
            (141, 14),
            (141, 15),
            (141, 16),
            (141, 17),
            (141, 18),
            (141, 19),
            (141, 20),
            (141, 21),
            (141, 22),
            (141, 23),
            (141, 24),
            (141, 25),
            (141, 26),
            (141, 27),
            (141, 28),
            (141, 29),
            (141, 30),
            (141, 31),
            (141, 32),
            (141, 33),
            (141, 34),
            (141, 35),
            (141, 36),
            (141, 37),
            (142, 0),
            (142, 1),
            (142, 2),
            (142, 3),
            (142, 4),
            (142, 5),
            (142, 6),
            (142, 7),
            (142, 8),
            (142, 9),
            (142, 10),
            (142, 11),
            (142, 12),
            (142, 13),
            (142, 14),
            (142, 15),
            (142, 16),
            (142, 17),
            (142, 18),
            (142, 19),
            (142, 20),
            (142, 21),
            (142, 22),
            (142, 23),
            (142, 24),
            (142, 25),
            (142, 26),
            (142, 27),
            (142, 28),
            (142, 29),
            (142, 30),
            (142, 31),
            (142, 32),
            (142, 33),
            (142, 34),
            (142, 35),
            (142, 36),
            (142, 37),
            (143, 0),
            (143, 1),
            (143, 2),
            (143, 3),
            (143, 4),
            (143, 5),
            (143, 6),
            (143, 7),
            (143, 8),
            (143, 9),
            (143, 10),
            (143, 11),
            (143, 12),
            (143, 13),
            (143, 14),
            (143, 15),
            (143, 16),
            (143, 17),
            (143, 18),
            (143, 19),
            (143, 20),
            (143, 21),
            (143, 22),
            (143, 23),
            (143, 24),
            (143, 25),
            (143, 26),
            (143, 27),
            (143, 28),
            (143, 29),
            (143, 30),
            (143, 31),
            (143, 32),
            (143, 33),
            (143, 34),
            (143, 35),
            (143, 36),
            (143, 37),
            (144, 0),
            (144, 1),
            (144, 2),
            (144, 3),
            (144, 4),
            (144, 5),
            (144, 6),
            (144, 7),
            (144, 8),
            (144, 9),
            (144, 10),
            (144, 11),
            (144, 12),
            (144, 13),
            (144, 14),
            (144, 15),
            (144, 16),
            (144, 17),
            (144, 18),
            (144, 19),
            (144, 20),
            (144, 21),
            (144, 22),
            (144, 23),
            (144, 24),
            (144, 25),
            (144, 26),
            (144, 27),
            (144, 28),
            (144, 29),
            (144, 30),
            (144, 31),
            (144, 32),
            (144, 33),
            (144, 34),
            (144, 35),
            (144, 36),
            (144, 37),
            (144, 38),
            (145, 0),
            (145, 1),
            (145, 2),
            (145, 3),
            (145, 4),
            (145, 5),
            (145, 6),
            (145, 7),
            (145, 8),
            (145, 9),
            (145, 10),
            (145, 11),
            (145, 12),
            (145, 13),
            (145, 14),
            (145, 15),
            (145, 16),
            (145, 17),
            (145, 18),
            (145, 19),
            (145, 20),
            (145, 21),
            (145, 22),
            (145, 23),
            (145, 24),
            (145, 25),
            (145, 26),
            (145, 27),
            (145, 28),
            (145, 29),
            (145, 30),
            (145, 31),
            (145, 32),
            (145, 33),
            (145, 34),
            (145, 35),
            (145, 36),
            (145, 37),
            (145, 38),
            (146, 0),
            (146, 1),
            (146, 2),
            (146, 3),
            (146, 4),
            (146, 5),
            (146, 6),
            (146, 7),
            (146, 8),
            (146, 9),
            (146, 10),
            (146, 11),
            (146, 12),
            (146, 13),
            (146, 14),
            (146, 15),
            (146, 16),
            (146, 17),
            (146, 18),
            (146, 19),
            (146, 20),
            (146, 21),
            (146, 22),
            (146, 23),
            (146, 24),
            (146, 25),
            (146, 26),
            (146, 27),
            (146, 28),
            (146, 29),
            (146, 30),
            (146, 31),
            (146, 32),
            (146, 33),
            (146, 34),
            (146, 35),
            (146, 36),
            (146, 37),
            (146, 38),
            (147, 0),
            (147, 1),
            (147, 2),
            (147, 3),
            (147, 4),
            (147, 5),
            (147, 6),
            (147, 7),
            (147, 8),
            (147, 9),
            (147, 10),
            (147, 11),
            (147, 12),
            (147, 13),
            (147, 14),
            (147, 15),
            (147, 16),
            (147, 17),
            (147, 18),
            (147, 19),
            (147, 20),
            (147, 21),
            (147, 22),
            (147, 23),
            (147, 24),
            (147, 25),
            (147, 26),
            (147, 27),
            (147, 28),
            (147, 29),
            (147, 30),
            (147, 31),
            (147, 32),
            (147, 33),
            (147, 34),
            (147, 35),
            (147, 36),
            (147, 37),
            (147, 38),
            (148, 0),
            (148, 1),
            (148, 2),
            (148, 3),
            (148, 4),
            (148, 5),
            (148, 6),
            (148, 7),
            (148, 8),
            (148, 9),
            (148, 10),
            (148, 11),
            (148, 12),
            (148, 13),
            (148, 14),
            (148, 15),
            (148, 16),
            (148, 17),
            (148, 18),
            (148, 19),
            (148, 20),
            (148, 21),
            (148, 22),
            (148, 23),
            (148, 24),
            (148, 25),
            (148, 26),
            (148, 27),
            (148, 28),
            (148, 29),
            (148, 30),
            (148, 31),
            (148, 32),
            (148, 33),
            (148, 34),
            (148, 35),
            (148, 36),
            (148, 37),
            (148, 38),
            (148, 39),
            (149, 0),
            (149, 1),
            (149, 2),
            (149, 3),
            (149, 4),
            (149, 5),
            (149, 6),
            (149, 7),
            (149, 8),
            (149, 9),
            (149, 10),
            (149, 11),
            (149, 12),
            (149, 13),
            (149, 14),
            (149, 15),
            (149, 16),
            (149, 17),
            (149, 18),
            (149, 19),
            (149, 20),
            (149, 21),
            (149, 22),
            (149, 23),
            (149, 24),
            (149, 25),
            (149, 26),
            (149, 27),
            (149, 28),
            (149, 29),
            (149, 30),
            (149, 31),
            (149, 32),
            (149, 33),
            (149, 34),
            (149, 35),
            (149, 36),
            (149, 37),
            (149, 38),
            (149, 39),
            (150, 0),
            (150, 1),
            (150, 2),
            (150, 3),
            (150, 4),
            (150, 5),
            (150, 6),
            (150, 7),
            (150, 8),
            (150, 9),
            (150, 10),
            (150, 11),
            (150, 12),
            (150, 13),
            (150, 14),
            (150, 15),
            (150, 16),
            (150, 17),
            (150, 18),
            (150, 19),
            (150, 20),
            (150, 21),
            (150, 22),
            (150, 23),
            (150, 24),
            (150, 25),
            (150, 26),
            (150, 27),
            (150, 28),
            (150, 29),
            (150, 30),
            (150, 31),
            (150, 32),
            (150, 33),
            (150, 34),
            (150, 35),
            (150, 36),
            (150, 37),
            (150, 38),
            (150, 39),
            (151, 0),
            (151, 1),
            (151, 2),
            (151, 3),
            (151, 4),
            (151, 5),
            (151, 6),
            (151, 7),
            (151, 8),
            (151, 9),
            (151, 10),
            (151, 11),
            (151, 12),
            (151, 13),
            (151, 14),
            (151, 15),
            (151, 16),
            (151, 17),
            (151, 18),
            (151, 19),
            (151, 20),
            (151, 21),
            (151, 22),
            (151, 23),
            (151, 24),
            (151, 25),
            (151, 26),
            (151, 27),
            (151, 28),
            (151, 29),
            (151, 30),
            (151, 31),
            (151, 32),
            (151, 33),
            (151, 34),
            (151, 35),
            (151, 36),
            (151, 37),
            (151, 38),
            (151, 39),
            (151, 40),
            (152, 0),
            (152, 1),
            (152, 2),
            (152, 3),
            (152, 4),
            (152, 5),
            (152, 6),
            (152, 7),
            (152, 8),
            (152, 9),
            (152, 10),
            (152, 11),
            (152, 12),
            (152, 13),
            (152, 14),
            (152, 15),
            (152, 16),
            (152, 17),
            (152, 18),
            (152, 19),
            (152, 20),
            (152, 21),
            (152, 22),
            (152, 23),
            (152, 24),
            (152, 25),
            (152, 26),
            (152, 27),
            (152, 28),
            (152, 29),
            (152, 30),
            (152, 31),
            (152, 32),
            (152, 33),
            (152, 34),
            (152, 35),
            (152, 36),
            (152, 37),
            (152, 38),
            (152, 39),
            (152, 40),
            (153, 0),
            (153, 1),
            (153, 2),
            (153, 3),
            (153, 4),
            (153, 5),
            (153, 6),
            (153, 7),
            (153, 8),
            (153, 9),
            (153, 10),
            (153, 11),
            (153, 12),
            (153, 13),
            (153, 14),
            (153, 15),
            (153, 16),
            (153, 17),
            (153, 18),
            (153, 19),
            (153, 20),
            (153, 21),
            (153, 22),
            (153, 23),
            (153, 24),
            (153, 25),
            (153, 26),
            (153, 27),
            (153, 28),
            (153, 29),
            (153, 30),
            (153, 31),
            (153, 32),
            (153, 33),
            (153, 34),
            (153, 35),
            (153, 36),
            (153, 37),
            (153, 38),
            (153, 39),
            (153, 40),
            (154, 0),
            (154, 1),
            (154, 2),
            (154, 3),
            (154, 4),
            (154, 5),
            (154, 6),
            (154, 7),
            (154, 8),
            (154, 9),
            (154, 10),
            (154, 11),
            (154, 12),
            (154, 13),
            (154, 14),
            (154, 15),
            (154, 16),
            (154, 17),
            (154, 18),
            (154, 19),
            (154, 20),
            (154, 21),
            (154, 22),
            (154, 23),
            (154, 24),
            (154, 25),
            (154, 26),
            (154, 27),
            (154, 28),
            (154, 29),
            (154, 30),
            (154, 31),
            (154, 32),
            (154, 33),
            (154, 34),
            (154, 35),
            (154, 36),
            (154, 37),
            (154, 38),
            (154, 39),
            (154, 40),
            (155, 0),
            (155, 1),
            (155, 2),
            (155, 3),
            (155, 4),
            (155, 5),
            (155, 6),
            (155, 7),
            (155, 8),
            (155, 9),
            (155, 10),
            (155, 11),
            (155, 12),
            (155, 13),
            (155, 14),
            (155, 15),
            (155, 16),
            (155, 17),
            (155, 18),
            (155, 19),
            (155, 20),
            (155, 21),
            (155, 22),
            (155, 23),
            (155, 24),
            (155, 25),
            (155, 26),
            (155, 27),
            (155, 28),
            (155, 29),
            (155, 30),
            (155, 31),
            (155, 32),
            (155, 33),
            (155, 34),
            (155, 35),
            (155, 36),
            (155, 37),
            (155, 38),
            (155, 39),
            (155, 40),
            (155, 41),
            (156, 0),
            (156, 1),
            (156, 2),
            (156, 3),
            (156, 4),
            (156, 5),
            (156, 6),
            (156, 7),
            (156, 8),
            (156, 9),
            (156, 10),
            (156, 11),
            (156, 12),
            (156, 13),
            (156, 14),
            (156, 15),
            (156, 16),
            (156, 17),
            (156, 18),
            (156, 19),
            (156, 20),
            (156, 21),
            (156, 22),
            (156, 23),
            (156, 24),
            (156, 25),
            (156, 26),
            (156, 27),
            (156, 28),
            (156, 29),
            (156, 30),
            (156, 31),
            (156, 32),
            (156, 33),
            (156, 34),
            (156, 35),
            (156, 36),
            (156, 37),
            (156, 38),
            (156, 39),
            (156, 40),
            (156, 41),
            (157, 0),
            (157, 1),
            (157, 2),
            (157, 3),
            (157, 4),
            (157, 5),
            (157, 6),
            (157, 7),
            (157, 8),
            (157, 9),
            (157, 10),
            (157, 11),
            (157, 12),
            (157, 13),
            (157, 14),
            (157, 15),
            (157, 16),
            (157, 17),
            (157, 18),
            (157, 19),
            (157, 20),
            (157, 21),
            (157, 22),
            (157, 23),
            (157, 24),
            (157, 25),
            (157, 26),
            (157, 27),
            (157, 28),
            (157, 29),
            (157, 30),
            (157, 31),
            (157, 32),
            (157, 33),
            (157, 34),
            (157, 35),
            (157, 36),
            (157, 37),
            (157, 38),
            (157, 39),
            (157, 40),
            (157, 41),
            (158, 0),
            (158, 1),
            (158, 2),
            (158, 3),
            (158, 4),
            (158, 5),
            (158, 6),
            (158, 7),
            (158, 8),
            (158, 9),
            (158, 10),
            (158, 11),
            (158, 12),
            (158, 13),
            (158, 14),
            (158, 15),
            (158, 16),
            (158, 17),
            (158, 18),
            (158, 19),
            (158, 20),
            (158, 21),
            (158, 22),
            (158, 23),
            (158, 24),
            (158, 25),
            (158, 26),
            (158, 27),
            (158, 28),
            (158, 29),
            (158, 30),
            (158, 31),
            (158, 32),
            (158, 33),
            (158, 34),
            (158, 35),
            (158, 36),
            (158, 37),
            (158, 38),
            (158, 39),
            (158, 40),
            (158, 41),
            (158, 42),
            (159, 0),
            (159, 1),
            (159, 2),
            (159, 3),
            (159, 4),
            (159, 5),
            (159, 6),
            (159, 7),
            (159, 8),
            (159, 9),
            (159, 10),
            (159, 11),
            (159, 12),
            (159, 13),
            (159, 14),
            (159, 15),
            (159, 16),
            (159, 17),
            (159, 18),
            (159, 19),
            (159, 20),
            (159, 21),
            (159, 22),
            (159, 23),
            (159, 24),
            (159, 25),
            (159, 26),
            (159, 27),
            (159, 28),
            (159, 29),
            (159, 30),
            (159, 31),
            (159, 32),
            (159, 33),
            (159, 34),
            (159, 35),
            (159, 36),
            (159, 37),
            (159, 38),
            (159, 39),
            (159, 40),
            (159, 41),
            (159, 42),
        }:
            return 22
        return 54

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"(", ")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 4

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_5_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"("}:
            return position == 5
        elif token in {")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 1

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_4_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 5
        elif attn_0_0_output in {")"}:
            return position == 9
        elif attn_0_0_output in {"<s>"}:
            return position == 2

    attn_2_2_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_5_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 3
        elif attn_0_0_output in {")"}:
            return position == 4
        elif attn_0_0_output in {"<s>"}:
            return position == 5

    attn_2_3_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_4_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 4
        elif attn_0_0_output in {")"}:
            return position == 5
        elif attn_0_0_output in {"<s>"}:
            return position == 1

    attn_2_4_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_3_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_5_output, attn_0_4_output):
        if attn_0_5_output in {"("}:
            return attn_0_4_output == ""
        elif attn_0_5_output in {")"}:
            return attn_0_4_output == ")"
        elif attn_0_5_output in {"<s>"}:
            return attn_0_4_output == "("

    attn_2_5_pattern = select_closest(attn_0_4_outputs, attn_0_5_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_7_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_0_output, position):
        if attn_0_0_output in {"(", "<s>"}:
            return position == 3
        elif attn_0_0_output in {")"}:
            return position == 5

    attn_2_6_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_5_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_4_output, position):
        if attn_0_4_output in {"("}:
            return position == 4
        elif attn_0_4_output in {")"}:
            return position == 7
        elif attn_0_4_output in {"<s>"}:
            return position == 2

    attn_2_7_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_1_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_1_1_output, mlp_0_0_output):
        if mlp_1_1_output in {0, 12, 39}:
            return mlp_0_0_output == 51
        elif mlp_1_1_output in {1, 66}:
            return mlp_0_0_output == 49
        elif mlp_1_1_output in {2, 43, 14, 46, 19, 31}:
            return mlp_0_0_output == 76
        elif mlp_1_1_output in {32, 64, 3, 36, 5, 9, 41, 20, 21, 53, 23, 24, 26, 62}:
            return mlp_0_0_output == 58
        elif mlp_1_1_output in {4}:
            return mlp_0_0_output == 11
        elif mlp_1_1_output in {6}:
            return mlp_0_0_output == 62
        elif mlp_1_1_output in {7}:
            return mlp_0_0_output == 71
        elif mlp_1_1_output in {8}:
            return mlp_0_0_output == 10
        elif mlp_1_1_output in {10}:
            return mlp_0_0_output == 26
        elif mlp_1_1_output in {11, 28}:
            return mlp_0_0_output == 66
        elif mlp_1_1_output in {35, 13}:
            return mlp_0_0_output == 47
        elif mlp_1_1_output in {15}:
            return mlp_0_0_output == 56
        elif mlp_1_1_output in {16}:
            return mlp_0_0_output == 8
        elif mlp_1_1_output in {17}:
            return mlp_0_0_output == 46
        elif mlp_1_1_output in {18}:
            return mlp_0_0_output == 9
        elif mlp_1_1_output in {70, 22}:
            return mlp_0_0_output == 32
        elif mlp_1_1_output in {25}:
            return mlp_0_0_output == 12
        elif mlp_1_1_output in {27}:
            return mlp_0_0_output == 18
        elif mlp_1_1_output in {29}:
            return mlp_0_0_output == 67
        elif mlp_1_1_output in {30}:
            return mlp_0_0_output == 36
        elif mlp_1_1_output in {33, 59}:
            return mlp_0_0_output == 69
        elif mlp_1_1_output in {72, 34}:
            return mlp_0_0_output == 40
        elif mlp_1_1_output in {40, 37, 71}:
            return mlp_0_0_output == 60
        elif mlp_1_1_output in {38}:
            return mlp_0_0_output == 2
        elif mlp_1_1_output in {42}:
            return mlp_0_0_output == 22
        elif mlp_1_1_output in {56, 44}:
            return mlp_0_0_output == 64
        elif mlp_1_1_output in {45}:
            return mlp_0_0_output == 34
        elif mlp_1_1_output in {47}:
            return mlp_0_0_output == 37
        elif mlp_1_1_output in {48}:
            return mlp_0_0_output == 61
        elif mlp_1_1_output in {49}:
            return mlp_0_0_output == 24
        elif mlp_1_1_output in {50}:
            return mlp_0_0_output == 43
        elif mlp_1_1_output in {65, 51}:
            return mlp_0_0_output == 35
        elif mlp_1_1_output in {52}:
            return mlp_0_0_output == 68
        elif mlp_1_1_output in {54}:
            return mlp_0_0_output == 78
        elif mlp_1_1_output in {55}:
            return mlp_0_0_output == 20
        elif mlp_1_1_output in {57}:
            return mlp_0_0_output == 54
        elif mlp_1_1_output in {58}:
            return mlp_0_0_output == 31
        elif mlp_1_1_output in {60}:
            return mlp_0_0_output == 70
        elif mlp_1_1_output in {69, 61}:
            return mlp_0_0_output == 30
        elif mlp_1_1_output in {63}:
            return mlp_0_0_output == 28
        elif mlp_1_1_output in {67}:
            return mlp_0_0_output == 29
        elif mlp_1_1_output in {68}:
            return mlp_0_0_output == 33
        elif mlp_1_1_output in {73}:
            return mlp_0_0_output == 74
        elif mlp_1_1_output in {74, 78}:
            return mlp_0_0_output == 57
        elif mlp_1_1_output in {75}:
            return mlp_0_0_output == 44
        elif mlp_1_1_output in {76}:
            return mlp_0_0_output == 48
        elif mlp_1_1_output in {77}:
            return mlp_0_0_output == 38
        elif mlp_1_1_output in {79}:
            return mlp_0_0_output == 4

    num_attn_2_0_pattern = select(mlp_0_0_outputs, mlp_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, mlp_0_1_output):
        if attn_1_1_output in {"(", "<s>", ")"}:
            return mlp_0_1_output == 68

    num_attn_2_1_pattern = select(mlp_0_1_outputs, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_7_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_4_output, token):
        if attn_0_4_output in {"(", "<s>", ")"}:
            return token == ""

    num_attn_2_2_pattern = select(tokens, attn_0_4_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_7_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_7_output, mlp_0_0_output):
        if attn_1_7_output in {"(", "<s>"}:
            return mlp_0_0_output == 39
        elif attn_1_7_output in {")"}:
            return mlp_0_0_output == 58

    num_attn_2_3_pattern = select(mlp_0_0_outputs, attn_1_7_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_6_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            1,
            2,
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
            78,
            79,
        }:
            return token == ""
        elif mlp_0_0_output in {59}:
            return token == "("
        elif mlp_0_0_output in {77}:
            return token == "<pad>"

    num_attn_2_4_pattern = select(tokens, mlp_0_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_1_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_6_output, attn_1_4_output):
        if attn_1_6_output in {"(", "<s>", ")"}:
            return attn_1_4_output == ""

    num_attn_2_5_pattern = select(attn_1_4_outputs, attn_1_6_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_0_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(position, mlp_0_0_output):
        if position in {0, 3, 4, 5, 70, 7, 8, 9, 10, 40, 45, 47}:
            return mlp_0_0_output == 39
        elif position in {1}:
            return mlp_0_0_output == 50
        elif position in {72, 2, 30}:
            return mlp_0_0_output == 14
        elif position in {64, 66, 6, 74, 11, 77, 51, 52, 55}:
            return mlp_0_0_output == 2
        elif position in {73, 12, 14, 16, 18, 20, 22, 24, 26, 59}:
            return mlp_0_0_output == 73
        elif position in {25, 13}:
            return mlp_0_0_output == 16
        elif position in {57, 15}:
            return mlp_0_0_output == 79
        elif position in {17}:
            return mlp_0_0_output == 55
        elif position in {19}:
            return mlp_0_0_output == 64
        elif position in {21}:
            return mlp_0_0_output == 33
        elif position in {23}:
            return mlp_0_0_output == 47
        elif position in {27, 76}:
            return mlp_0_0_output == 77
        elif position in {28}:
            return mlp_0_0_output == 60
        elif position in {33, 29, 46}:
            return mlp_0_0_output == 20
        elif position in {31}:
            return mlp_0_0_output == 9
        elif position in {32}:
            return mlp_0_0_output == 23
        elif position in {34, 67}:
            return mlp_0_0_output == 75
        elif position in {35, 60}:
            return mlp_0_0_output == 43
        elif position in {50, 36}:
            return mlp_0_0_output == 28
        elif position in {37}:
            return mlp_0_0_output == 54
        elif position in {43, 38}:
            return mlp_0_0_output == 56
        elif position in {75, 39}:
            return mlp_0_0_output == 25
        elif position in {41}:
            return mlp_0_0_output == 40
        elif position in {42, 54}:
            return mlp_0_0_output == 70
        elif position in {44}:
            return mlp_0_0_output == 29
        elif position in {48, 71}:
            return mlp_0_0_output == 59
        elif position in {56, 49}:
            return mlp_0_0_output == 58
        elif position in {53}:
            return mlp_0_0_output == 48
        elif position in {58, 78}:
            return mlp_0_0_output == 49
        elif position in {61}:
            return mlp_0_0_output == 46
        elif position in {62}:
            return mlp_0_0_output == 62
        elif position in {63}:
            return mlp_0_0_output == 53
        elif position in {65}:
            return mlp_0_0_output == 65
        elif position in {68}:
            return mlp_0_0_output == 27
        elif position in {69}:
            return mlp_0_0_output == 17
        elif position in {79}:
            return mlp_0_0_output == 3

    num_attn_2_6_pattern = select(mlp_0_0_outputs, positions, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_2_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_0_output, attn_1_6_output):
        if attn_1_0_output in {"(", "<s>", ")"}:
            return attn_1_6_output == ""

    num_attn_2_7_pattern = select(attn_1_6_outputs, attn_1_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_1_3_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_0_output, attn_0_1_output):
        key = (attn_1_0_output, attn_0_1_output)
        return 77

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_0_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_5_output, attn_2_7_output):
        key = (attn_2_5_output, attn_2_7_output)
        if key in {("(", "<s>"), ("<s>", "<s>")}:
            return 25
        return 38

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_5_outputs, attn_2_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output, num_attn_2_2_output):
        key = (num_attn_1_0_output, num_attn_2_2_output)
        return 52

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_6_output, num_attn_2_0_output):
        key = (num_attn_1_6_output, num_attn_2_0_output)
        return 35

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_2_0_outputs)
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
            "(",
            ")",
            ")",
            "(",
            "(",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            ")",
            "(",
            "(",
            "(",
            "(",
            ")",
            ")",
            "(",
            "(",
            "(",
            ")",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            ")",
            ")",
            "(",
            ")",
            "(",
            "(",
            ")",
        ]
    )
)
