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
        "output/length/rasp/dyck1/trainlength40/s4/dyck1_weights.csv",
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
            return position == 5
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 3

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {
            0,
            1,
            2,
            40,
            42,
            45,
            47,
            49,
            50,
            51,
            52,
            56,
            59,
            61,
            62,
            63,
            65,
            68,
            69,
            72,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return k_position == 1
        elif q_position in {3, 4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {29, 6, 31}:
            return k_position == 4
        elif q_position in {66, 7, 8, 41, 44, 53, 57, 58}:
            return k_position == 7
        elif q_position in {67, 70, 9, 17, 27, 28}:
            return k_position == 8
        elif q_position in {33, 10, 60}:
            return k_position == 5
        elif q_position in {11, 13}:
            return k_position == 9
        elif q_position in {16, 12, 30}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {34, 15}:
            return k_position == 11
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 15
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 18
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25, 37, 39}:
            return k_position == 6
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {32}:
            return k_position == 31
        elif q_position in {35}:
            return k_position == 28
        elif q_position in {36}:
            return k_position == 35
        elif q_position in {38}:
            return k_position == 37
        elif q_position in {43}:
            return k_position == 32
        elif q_position in {46}:
            return k_position == 40
        elif q_position in {48}:
            return k_position == 44
        elif q_position in {54}:
            return k_position == 52
        elif q_position in {55}:
            return k_position == 38
        elif q_position in {64}:
            return k_position == 41
        elif q_position in {71}:
            return k_position == 66
        elif q_position in {73}:
            return k_position == 20

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 17}:
            return k_position == 7
        elif q_position in {1, 50, 12, 13}:
            return k_position == 11
        elif q_position in {8, 2}:
            return k_position == 6
        elif q_position in {3, 6}:
            return k_position == 2
        elif q_position in {4, 39}:
            return k_position == 3
        elif q_position in {33, 5, 47}:
            return k_position == 32
        elif q_position in {7}:
            return k_position == 44
        elif q_position in {9, 14}:
            return k_position == 12
        elif q_position in {10, 15}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 9
        elif q_position in {16, 65, 58, 51}:
            return k_position == 15
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {24, 19, 21}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {27, 77, 22}:
            return k_position == 21
        elif q_position in {56, 25, 23}:
            return k_position == 22
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {32, 29}:
            return k_position == 28
        elif q_position in {30}:
            return k_position == 29
        elif q_position in {38, 31}:
            return k_position == 30
        elif q_position in {34, 36}:
            return k_position == 26
        elif q_position in {35, 68, 37, 69, 43, 46}:
            return k_position == 5
        elif q_position in {40}:
            return k_position == 64
        elif q_position in {41}:
            return k_position == 49
        elif q_position in {42}:
            return k_position == 52
        elif q_position in {44}:
            return k_position == 74
        elif q_position in {45, 79}:
            return k_position == 13
        elif q_position in {48}:
            return k_position == 67
        elif q_position in {49}:
            return k_position == 48
        elif q_position in {52, 70}:
            return k_position == 57
        elif q_position in {73, 57, 53}:
            return k_position == 72
        elif q_position in {54}:
            return k_position == 51
        elif q_position in {67, 55}:
            return k_position == 78
        elif q_position in {59}:
            return k_position == 54
        elif q_position in {60}:
            return k_position == 77
        elif q_position in {61, 63}:
            return k_position == 50
        elif q_position in {62}:
            return k_position == 61
        elif q_position in {64}:
            return k_position == 47
        elif q_position in {66}:
            return k_position == 34
        elif q_position in {71}:
            return k_position == 76
        elif q_position in {72}:
            return k_position == 73
        elif q_position in {74}:
            return k_position == 33
        elif q_position in {75, 78}:
            return k_position == 68
        elif q_position in {76}:
            return k_position == 71

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 66, 36, 40, 75, 12, 14, 17, 30}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {48, 3}:
            return k_position == 33
        elif q_position in {4, 39}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 51
        elif q_position in {33, 35, 37, 6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 45
        elif q_position in {67, 8, 42, 74, 45, 62}:
            return k_position == 7
        elif q_position in {9, 11, 47}:
            return k_position == 55
        elif q_position in {25, 10, 18}:
            return k_position == 8
        elif q_position in {13, 15}:
            return k_position == 60
        elif q_position in {16}:
            return k_position == 14
        elif q_position in {19, 38}:
            return k_position == 12
        elif q_position in {20, 21}:
            return k_position == 15
        elif q_position in {22}:
            return k_position == 18
        elif q_position in {60, 23}:
            return k_position == 76
        elif q_position in {24, 28}:
            return k_position == 9
        elif q_position in {26}:
            return k_position == 17
        elif q_position in {27}:
            return k_position == 54
        elif q_position in {65, 29}:
            return k_position == 59
        elif q_position in {76, 61, 31}:
            return k_position == 49
        elif q_position in {32}:
            return k_position == 26
        elif q_position in {34}:
            return k_position == 10
        elif q_position in {72, 41, 73}:
            return k_position == 23
        elif q_position in {43, 68, 55}:
            return k_position == 21
        elif q_position in {44}:
            return k_position == 53
        elif q_position in {46, 71}:
            return k_position == 50
        elif q_position in {49, 53}:
            return k_position == 78
        elif q_position in {50}:
            return k_position == 70
        elif q_position in {51}:
            return k_position == 35
        elif q_position in {52, 69}:
            return k_position == 48
        elif q_position in {56, 54}:
            return k_position == 16
        elif q_position in {57, 63}:
            return k_position == 31
        elif q_position in {64, 58}:
            return k_position == 69
        elif q_position in {59}:
            return k_position == 56
        elif q_position in {70}:
            return k_position == 29
        elif q_position in {77, 78}:
            return k_position == 13
        elif q_position in {79}:
            return k_position == 68

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
            return position == 47

    attn_0_4_pattern = select_closest(positions, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 2

    attn_0_5_pattern = select_closest(positions, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 55

    attn_0_6_pattern = select_closest(positions, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 66
        elif q_position in {1, 9}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 30
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {16, 10, 27, 70}:
            return k_position == 9
        elif q_position in {51, 11, 60}:
            return k_position == 10
        elif q_position in {41, 34, 12, 46}:
            return k_position == 11
        elif q_position in {56, 13, 71, 55}:
            return k_position == 12
        elif q_position in {29, 14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {17, 19, 21}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {61, 22}:
            return k_position == 21
        elif q_position in {68, 23}:
            return k_position == 20
        elif q_position in {24, 63}:
            return k_position == 23
        elif q_position in {25, 77, 54, 78}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {28, 52}:
            return k_position == 27
        elif q_position in {69, 30}:
            return k_position == 29
        elif q_position in {64, 53, 38, 31}:
            return k_position == 37
        elif q_position in {32}:
            return k_position == 31
        elif q_position in {33}:
            return k_position == 32
        elif q_position in {35}:
            return k_position == 34
        elif q_position in {36}:
            return k_position == 35
        elif q_position in {50, 67, 37}:
            return k_position == 36
        elif q_position in {39}:
            return k_position == 55
        elif q_position in {40}:
            return k_position == 59
        elif q_position in {49, 42}:
            return k_position == 65
        elif q_position in {43}:
            return k_position == 18
        elif q_position in {44}:
            return k_position == 46
        elif q_position in {45}:
            return k_position == 60
        elif q_position in {47}:
            return k_position == 56
        elif q_position in {48}:
            return k_position == 28
        elif q_position in {57}:
            return k_position == 57
        elif q_position in {58}:
            return k_position == 15
        elif q_position in {73, 59}:
            return k_position == 40
        elif q_position in {62, 79}:
            return k_position == 78
        elif q_position in {65}:
            return k_position == 47
        elif q_position in {66}:
            return k_position == 42
        elif q_position in {72}:
            return k_position == 41
        elif q_position in {74}:
            return k_position == 58
        elif q_position in {75}:
            return k_position == 54
        elif q_position in {76}:
            return k_position == 70

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {
            0,
            2,
            8,
            9,
            12,
            14,
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
        elif position in {1, 10, 11, 13, 15, 16, 17}:
            return token == "<s>"
        elif position in {3, 4, 5, 6, 7}:
            return token == ")"
        elif position in {37}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 20, 62}:
            return k_position == 45
        elif q_position in {1}:
            return k_position == 36
        elif q_position in {2, 42}:
            return k_position == 54
        elif q_position in {3, 71}:
            return k_position == 77
        elif q_position in {4}:
            return k_position == 16
        elif q_position in {69, 5, 16, 52, 24}:
            return k_position == 70
        elif q_position in {6}:
            return k_position == 19
        elif q_position in {7}:
            return k_position == 69
        elif q_position in {8, 25, 39, 23}:
            return k_position == 79
        elif q_position in {9}:
            return k_position == 26
        elif q_position in {10, 21}:
            return k_position == 32
        elif q_position in {26, 11}:
            return k_position == 46
        elif q_position in {57, 12, 77}:
            return k_position == 47
        elif q_position in {13, 46}:
            return k_position == 55
        elif q_position in {18, 14}:
            return k_position == 22
        elif q_position in {47, 15}:
            return k_position == 50
        elif q_position in {17}:
            return k_position == 71
        elif q_position in {19}:
            return k_position == 53
        elif q_position in {22, 79}:
            return k_position == 38
        elif q_position in {27}:
            return k_position == 56
        elif q_position in {40, 28}:
            return k_position == 59
        elif q_position in {58, 29}:
            return k_position == 51
        elif q_position in {30}:
            return k_position == 72
        elif q_position in {56, 43, 31}:
            return k_position == 35
        elif q_position in {32, 70}:
            return k_position == 41
        elif q_position in {33}:
            return k_position == 57
        elif q_position in {41, 34}:
            return k_position == 76
        elif q_position in {35, 45, 54}:
            return k_position == 64
        elif q_position in {67, 36}:
            return k_position == 78
        elif q_position in {37}:
            return k_position == 44
        elif q_position in {53, 38}:
            return k_position == 42
        elif q_position in {44}:
            return k_position == 66
        elif q_position in {48, 72, 63}:
            return k_position == 61
        elif q_position in {49}:
            return k_position == 37
        elif q_position in {73, 50, 60}:
            return k_position == 58
        elif q_position in {51}:
            return k_position == 48
        elif q_position in {55}:
            return k_position == 73
        elif q_position in {59}:
            return k_position == 75
        elif q_position in {61}:
            return k_position == 62
        elif q_position in {64, 78}:
            return k_position == 63
        elif q_position in {65, 75}:
            return k_position == 40
        elif q_position in {66}:
            return k_position == 43
        elif q_position in {68}:
            return k_position == 65
        elif q_position in {74}:
            return k_position == 67
        elif q_position in {76}:
            return k_position == 39

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"(", ")", "<s>"}:
            return k_token == ""

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1, 28, 70}:
            return k_position == 34
        elif q_position in {2, 46}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 33
        elif q_position in {11, 4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 28
        elif q_position in {6}:
            return k_position == 37
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8, 18, 54}:
            return k_position == 60
        elif q_position in {9, 47}:
            return k_position == 13
        elif q_position in {17, 10}:
            return k_position == 54
        elif q_position in {12}:
            return k_position == 62
        elif q_position in {13}:
            return k_position == 64
        elif q_position in {69, 14}:
            return k_position == 73
        elif q_position in {15}:
            return k_position == 59
        elif q_position in {16, 34}:
            return k_position == 70
        elif q_position in {19}:
            return k_position == 29
        elif q_position in {20}:
            return k_position == 76
        elif q_position in {44, 21}:
            return k_position == 44
        elif q_position in {22}:
            return k_position == 52
        elif q_position in {23}:
            return k_position == 69
        elif q_position in {24, 41}:
            return k_position == 65
        elif q_position in {25}:
            return k_position == 35
        elif q_position in {26, 30}:
            return k_position == 68
        elif q_position in {40, 27}:
            return k_position == 56
        elif q_position in {29}:
            return k_position == 40
        elif q_position in {31}:
            return k_position == 43
        elif q_position in {32}:
            return k_position == 42
        elif q_position in {48, 33, 53}:
            return k_position == 63
        elif q_position in {50, 35}:
            return k_position == 12
        elif q_position in {72, 57, 43, 36}:
            return k_position == 66
        elif q_position in {37}:
            return k_position == 50
        elif q_position in {56, 38}:
            return k_position == 71
        elif q_position in {51, 39}:
            return k_position == 45
        elif q_position in {42, 52}:
            return k_position == 58
        elif q_position in {45, 55}:
            return k_position == 72
        elif q_position in {49, 60}:
            return k_position == 21
        elif q_position in {58}:
            return k_position == 1
        elif q_position in {65, 59}:
            return k_position == 39
        elif q_position in {61}:
            return k_position == 24
        elif q_position in {62}:
            return k_position == 23
        elif q_position in {63}:
            return k_position == 75
        elif q_position in {64}:
            return k_position == 46
        elif q_position in {66}:
            return k_position == 17
        elif q_position in {67}:
            return k_position == 14
        elif q_position in {74, 68}:
            return k_position == 15
        elif q_position in {73, 71}:
            return k_position == 0
        elif q_position in {75}:
            return k_position == 18
        elif q_position in {76}:
            return k_position == 78
        elif q_position in {77}:
            return k_position == 22
        elif q_position in {78}:
            return k_position == 51
        elif q_position in {79}:
            return k_position == 19

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"("}:
            return position == 12
        elif token in {")"}:
            return position == 50
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"("}:
            return position == 32
        elif token in {")"}:
            return position == 43
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"("}:
            return position == 75
        elif token in {")"}:
            return position == 0
        elif token in {"<s>"}:
            return position == 68

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(q_position, k_position):
        if q_position in {0, 36}:
            return k_position == 57
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 35}:
            return k_position == 65
        elif q_position in {3}:
            return k_position == 0
        elif q_position in {33, 4, 68, 46, 53, 59}:
            return k_position == 39
        elif q_position in {41, 26, 5, 70}:
            return k_position == 58
        elif q_position in {6}:
            return k_position == 75
        elif q_position in {66, 7}:
            return k_position == 47
        elif q_position in {8}:
            return k_position == 63
        elif q_position in {9, 69}:
            return k_position == 70
        elif q_position in {10, 77, 31}:
            return k_position == 49
        elif q_position in {11}:
            return k_position == 34
        elif q_position in {12, 61}:
            return k_position == 72
        elif q_position in {43, 13}:
            return k_position == 64
        elif q_position in {14}:
            return k_position == 32
        elif q_position in {15}:
            return k_position == 41
        elif q_position in {16}:
            return k_position == 55
        elif q_position in {17, 44}:
            return k_position == 52
        elif q_position in {24, 18, 51, 34}:
            return k_position == 51
        elif q_position in {19}:
            return k_position == 33
        elif q_position in {20}:
            return k_position == 37
        elif q_position in {21}:
            return k_position == 21
        elif q_position in {22}:
            return k_position == 60
        elif q_position in {54, 23}:
            return k_position == 59
        elif q_position in {25, 58, 74, 47}:
            return k_position == 53
        elif q_position in {42, 27}:
            return k_position == 62
        elif q_position in {28}:
            return k_position == 38
        elif q_position in {29}:
            return k_position == 9
        elif q_position in {32, 57, 30}:
            return k_position == 79
        elif q_position in {76, 37}:
            return k_position == 77
        elif q_position in {38}:
            return k_position == 66
        elif q_position in {40, 39}:
            return k_position == 69
        elif q_position in {71, 73, 45, 52, 63}:
            return k_position == 1
        elif q_position in {48, 62, 55}:
            return k_position == 71
        elif q_position in {49}:
            return k_position == 67
        elif q_position in {56, 50, 72}:
            return k_position == 78
        elif q_position in {60, 78}:
            return k_position == 54
        elif q_position in {64}:
            return k_position == 42
        elif q_position in {65}:
            return k_position == 76
        elif q_position in {67}:
            return k_position == 43
        elif q_position in {75}:
            return k_position == 61
        elif q_position in {79}:
            return k_position == 74

    num_attn_0_7_pattern = select(positions, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_7_output):
        key = (attn_0_1_output, attn_0_7_output)
        if key in {("(", "("), ("<s>", "(")}:
            return 28
        elif key in {("(", ")"), ("<s>", ")")}:
            return 27
        elif key in {(")", ")"), (")", "<s>")}:
            return 2
        return 51

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_7_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, token):
        key = (position, token)
        if key in {
            (0, "("),
            (1, "("),
            (2, "("),
            (5, "("),
            (5, "<s>"),
            (7, "("),
            (7, "<s>"),
            (13, "("),
            (15, "("),
            (15, "<s>"),
            (16, "("),
            (16, "<s>"),
            (17, "("),
            (17, "<s>"),
            (19, "("),
            (21, "("),
            (21, "<s>"),
            (22, "("),
            (22, "<s>"),
            (23, "("),
            (23, "<s>"),
            (25, "("),
            (25, ")"),
            (25, "<s>"),
            (27, "("),
            (27, ")"),
            (27, "<s>"),
            (29, "("),
            (29, ")"),
            (29, "<s>"),
            (31, "("),
            (31, "<s>"),
            (33, "("),
            (33, ")"),
            (33, "<s>"),
            (35, "("),
            (35, "<s>"),
            (36, "("),
            (37, "("),
            (37, ")"),
            (37, "<s>"),
            (38, "("),
            (38, "<s>"),
            (39, "("),
            (39, ")"),
            (39, "<s>"),
            (40, "("),
            (41, "("),
            (42, "("),
            (44, "("),
            (45, "("),
            (46, "("),
            (47, "("),
            (49, "("),
            (50, "("),
            (52, "("),
            (53, "("),
            (55, "("),
            (56, "("),
            (58, "("),
            (60, "("),
            (61, "("),
            (62, "("),
            (63, "("),
            (64, "("),
            (66, "("),
            (67, "("),
            (70, "("),
            (71, "("),
            (72, "("),
            (73, "("),
            (74, "("),
            (75, "("),
            (76, "("),
            (77, "("),
            (78, "("),
            (79, "("),
        }:
            return 61
        elif key in {
            (13, "<s>"),
            (19, "<s>"),
            (51, "("),
            (59, "("),
            (68, "("),
            (69, "("),
        }:
            return 76
        return 19

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_4_output, num_attn_0_7_output):
        key = (num_attn_0_4_output, num_attn_0_7_output)
        return 0

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_5_output):
        key = (num_attn_0_1_output, num_attn_0_5_output)
        return 75

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_4_output, position):
        if attn_0_4_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_4_output in {")"}:
            return position == 13

    attn_1_0_pattern = select_closest(positions, attn_0_4_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_4_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"("}:
            return position == 3
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 60

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_6_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, position):
        if token in {"("}:
            return position == 5
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 57

    attn_1_2_pattern = select_closest(positions, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 5

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(token, position):
        if token in {"("}:
            return position == 3
        elif token in {")"}:
            return position == 5
        elif token in {"<s>"}:
            return position == 1

    attn_1_4_pattern = select_closest(positions, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_4_output, position):
        if attn_0_4_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_4_output in {")"}:
            return position == 9

    attn_1_5_pattern = select_closest(positions, attn_0_4_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_1_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")", "<s>"}:
            return position == 5

    attn_1_6_pattern = select_closest(positions, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_4_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_5_output, position):
        if attn_0_5_output in {"("}:
            return position == 1
        elif attn_0_5_output in {")"}:
            return position == 12
        elif attn_0_5_output in {"<s>"}:
            return position == 3

    attn_1_7_pattern = select_closest(positions, attn_0_5_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_5_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_0_output, attn_0_6_output):
        if num_mlp_0_0_output in {0, 3, 4, 5, 68, 7, 8, 12, 76, 48, 57, 61}:
            return attn_0_6_output == ")"
        elif num_mlp_0_0_output in {
            1,
            2,
            6,
            9,
            10,
            11,
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
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            58,
            59,
            60,
            62,
            63,
            64,
            65,
            66,
            67,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            77,
            78,
            79,
        }:
            return attn_0_6_output == ""
        elif num_mlp_0_0_output in {35}:
            return attn_0_6_output == "<s>"

    num_attn_1_0_pattern = select(
        attn_0_6_outputs, num_mlp_0_0_outputs, num_predicate_1_0
    )
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_0_output, attn_0_5_output):
        if mlp_0_0_output in {
            0,
            1,
            2,
            3,
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
            46,
            47,
            48,
            49,
            50,
            51,
            52,
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
            75,
            76,
            77,
            78,
        }:
            return attn_0_5_output == ""
        elif mlp_0_0_output in {4, 74, 45, 79, 53}:
            return attn_0_5_output == "("

    num_attn_1_1_pattern = select(attn_0_5_outputs, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_4_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, mlp_0_0_output):
        if position in {0}:
            return mlp_0_0_output == 60
        elif position in {1, 13}:
            return mlp_0_0_output == 12
        elif position in {2, 12, 38}:
            return mlp_0_0_output == 13
        elif position in {11, 10, 3}:
            return mlp_0_0_output == 10
        elif position in {66, 4, 6, 71, 46, 14, 58, 60}:
            return mlp_0_0_output == 35
        elif position in {17, 5}:
            return mlp_0_0_output == 27
        elif position in {76, 7}:
            return mlp_0_0_output == 46
        elif position in {8, 21}:
            return mlp_0_0_output == 70
        elif position in {9}:
            return mlp_0_0_output == 15
        elif position in {15}:
            return mlp_0_0_output == 75
        elif position in {16, 32, 67}:
            return mlp_0_0_output == 72
        elif position in {18}:
            return mlp_0_0_output == 18
        elif position in {19}:
            return mlp_0_0_output == 54
        elif position in {20, 47}:
            return mlp_0_0_output == 36
        elif position in {22}:
            return mlp_0_0_output == 38
        elif position in {56, 23}:
            return mlp_0_0_output == 43
        elif position in {24, 59}:
            return mlp_0_0_output == 59
        elif position in {25, 78}:
            return mlp_0_0_output == 63
        elif position in {26}:
            return mlp_0_0_output == 9
        elif position in {51, 27}:
            return mlp_0_0_output == 53
        elif position in {28}:
            return mlp_0_0_output == 66
        elif position in {29, 55}:
            return mlp_0_0_output == 20
        elif position in {54, 63, 30, 39}:
            return mlp_0_0_output == 7
        elif position in {48, 31}:
            return mlp_0_0_output == 14
        elif position in {33}:
            return mlp_0_0_output == 17
        elif position in {34}:
            return mlp_0_0_output == 30
        elif position in {35}:
            return mlp_0_0_output == 26
        elif position in {36}:
            return mlp_0_0_output == 23
        elif position in {49, 44, 37}:
            return mlp_0_0_output == 68
        elif position in {40}:
            return mlp_0_0_output == 11
        elif position in {41}:
            return mlp_0_0_output == 47
        elif position in {42}:
            return mlp_0_0_output == 31
        elif position in {43}:
            return mlp_0_0_output == 57
        elif position in {45, 79}:
            return mlp_0_0_output == 73
        elif position in {50}:
            return mlp_0_0_output == 37
        elif position in {52, 77, 70}:
            return mlp_0_0_output == 55
        elif position in {53}:
            return mlp_0_0_output == 1
        elif position in {57}:
            return mlp_0_0_output == 64
        elif position in {73, 61}:
            return mlp_0_0_output == 33
        elif position in {62}:
            return mlp_0_0_output == 45
        elif position in {64, 68}:
            return mlp_0_0_output == 25
        elif position in {65}:
            return mlp_0_0_output == 74
        elif position in {69}:
            return mlp_0_0_output == 56
        elif position in {72}:
            return mlp_0_0_output == 58
        elif position in {74}:
            return mlp_0_0_output == 21
        elif position in {75}:
            return mlp_0_0_output == 16

    num_attn_1_2_pattern = select(mlp_0_0_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_4_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_3_output, mlp_0_0_output):
        if attn_0_3_output in {"("}:
            return mlp_0_0_output == 51
        elif attn_0_3_output in {")", "<s>"}:
            return mlp_0_0_output == 75

    num_attn_1_3_pattern = select(mlp_0_0_outputs, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_6_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(mlp_0_1_output, mlp_0_0_output):
        if mlp_0_1_output in {
            0,
            1,
            3,
            6,
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            19,
            20,
            21,
            23,
            24,
            26,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            38,
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
            54,
            55,
            56,
            57,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            68,
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
            return mlp_0_0_output == 2
        elif mlp_0_1_output in {2, 22, 39}:
            return mlp_0_0_output == 51
        elif mlp_0_1_output in {4}:
            return mlp_0_0_output == 56
        elif mlp_0_1_output in {66, 5, 70, 17, 58}:
            return mlp_0_0_output == 75
        elif mlp_0_1_output in {10, 53}:
            return mlp_0_0_output == 78
        elif mlp_0_1_output in {25, 18}:
            return mlp_0_0_output == 34
        elif mlp_0_1_output in {28}:
            return mlp_0_0_output == 32
        elif mlp_0_1_output in {37}:
            return mlp_0_0_output == 77
        elif mlp_0_1_output in {69}:
            return mlp_0_0_output == 59

    num_attn_1_4_pattern = select(mlp_0_0_outputs, mlp_0_1_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_6_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(mlp_0_0_output, num_mlp_0_1_output):
        if mlp_0_0_output in {0, 60}:
            return num_mlp_0_1_output == 48
        elif mlp_0_0_output in {1, 69, 70, 71, 11, 43, 61}:
            return num_mlp_0_1_output == 28
        elif mlp_0_0_output in {2}:
            return num_mlp_0_1_output == 5
        elif mlp_0_0_output in {3}:
            return num_mlp_0_1_output == 11
        elif mlp_0_0_output in {18, 4}:
            return num_mlp_0_1_output == 67
        elif mlp_0_0_output in {5, 7, 9, 10, 41, 13, 77, 20, 52}:
            return num_mlp_0_1_output == 32
        elif mlp_0_0_output in {17, 6, 49}:
            return num_mlp_0_1_output == 62
        elif mlp_0_0_output in {8, 31}:
            return num_mlp_0_1_output == 54
        elif mlp_0_0_output in {12}:
            return num_mlp_0_1_output == 39
        elif mlp_0_0_output in {76, 14, 25, 58, 62}:
            return num_mlp_0_1_output == 77
        elif mlp_0_0_output in {15}:
            return num_mlp_0_1_output == 41
        elif mlp_0_0_output in {16, 35}:
            return num_mlp_0_1_output == 50
        elif mlp_0_0_output in {73, 66, 19}:
            return num_mlp_0_1_output == 73
        elif mlp_0_0_output in {21}:
            return num_mlp_0_1_output == 4
        elif mlp_0_0_output in {22}:
            return num_mlp_0_1_output == 7
        elif mlp_0_0_output in {37, 23}:
            return num_mlp_0_1_output == 66
        elif mlp_0_0_output in {24}:
            return num_mlp_0_1_output == 53
        elif mlp_0_0_output in {26, 63}:
            return num_mlp_0_1_output == 47
        elif mlp_0_0_output in {27}:
            return num_mlp_0_1_output == 24
        elif mlp_0_0_output in {28}:
            return num_mlp_0_1_output == 60
        elif mlp_0_0_output in {29, 39}:
            return num_mlp_0_1_output == 26
        elif mlp_0_0_output in {30}:
            return num_mlp_0_1_output == 16
        elif mlp_0_0_output in {32}:
            return num_mlp_0_1_output == 38
        elif mlp_0_0_output in {33}:
            return num_mlp_0_1_output == 64
        elif mlp_0_0_output in {34, 53}:
            return num_mlp_0_1_output == 56
        elif mlp_0_0_output in {36}:
            return num_mlp_0_1_output == 12
        elif mlp_0_0_output in {78, 38}:
            return num_mlp_0_1_output == 36
        elif mlp_0_0_output in {40}:
            return num_mlp_0_1_output == 34
        elif mlp_0_0_output in {42}:
            return num_mlp_0_1_output == 63
        elif mlp_0_0_output in {44}:
            return num_mlp_0_1_output == 30
        elif mlp_0_0_output in {45}:
            return num_mlp_0_1_output == 59
        elif mlp_0_0_output in {46}:
            return num_mlp_0_1_output == 0
        elif mlp_0_0_output in {47}:
            return num_mlp_0_1_output == 76
        elif mlp_0_0_output in {48, 74, 67}:
            return num_mlp_0_1_output == 20
        elif mlp_0_0_output in {50}:
            return num_mlp_0_1_output == 43
        elif mlp_0_0_output in {51, 68}:
            return num_mlp_0_1_output == 15
        elif mlp_0_0_output in {54}:
            return num_mlp_0_1_output == 19
        elif mlp_0_0_output in {57, 55}:
            return num_mlp_0_1_output == 57
        elif mlp_0_0_output in {56}:
            return num_mlp_0_1_output == 25
        elif mlp_0_0_output in {59}:
            return num_mlp_0_1_output == 55
        elif mlp_0_0_output in {64}:
            return num_mlp_0_1_output == 1
        elif mlp_0_0_output in {65}:
            return num_mlp_0_1_output == 37
        elif mlp_0_0_output in {72}:
            return num_mlp_0_1_output == 61
        elif mlp_0_0_output in {75}:
            return num_mlp_0_1_output == 21
        elif mlp_0_0_output in {79}:
            return num_mlp_0_1_output == 49

    num_attn_1_5_pattern = select(
        num_mlp_0_1_outputs, mlp_0_0_outputs, num_predicate_1_5
    )
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_4_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(mlp_0_0_output, attn_0_2_output):
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
            return attn_0_2_output == ""

    num_attn_1_6_pattern = select(attn_0_2_outputs, mlp_0_0_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_2_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_3_output, mlp_0_0_output):
        if attn_0_3_output in {"("}:
            return mlp_0_0_output == 20
        elif attn_0_3_output in {")"}:
            return mlp_0_0_output == 32
        elif attn_0_3_output in {"<s>"}:
            return mlp_0_0_output == 40

    num_attn_1_7_pattern = select(mlp_0_0_outputs, attn_0_3_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {
            ("(", ")"),
            ("(", "<s>"),
            (")", ")"),
            (")", "<s>"),
            ("<s>", ")"),
            ("<s>", "<s>"),
        }:
            return 2
        return 20

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_6_output, attn_1_1_output):
        key = (attn_1_6_output, attn_1_1_output)
        if key in {("(", "(")}:
            return 5
        return 2

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_1_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_6_output, num_attn_1_7_output):
        key = (num_attn_1_6_output, num_attn_1_7_output)
        return 58

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_5_output, num_attn_1_7_output):
        key = (num_attn_1_5_output, num_attn_1_7_output)
        return 30

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_3_output, mlp_0_0_output):
        if attn_0_3_output in {"(", ")"}:
            return mlp_0_0_output == 2
        elif attn_0_3_output in {"<s>"}:
            return mlp_0_0_output == 70

    attn_2_0_pattern = select_closest(mlp_0_0_outputs, attn_0_3_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_4_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, position):
        if attn_0_1_output in {"("}:
            return position == 3
        elif attn_0_1_output in {")"}:
            return position == 9
        elif attn_0_1_output in {"<s>"}:
            return position == 1

    attn_2_1_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_6_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_5_output, position):
        if attn_0_5_output in {"("}:
            return position == 3
        elif attn_0_5_output in {")"}:
            return position == 11
        elif attn_0_5_output in {"<s>"}:
            return position == 1

    attn_2_2_pattern = select_closest(positions, attn_0_5_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_7_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_5_output, position):
        if attn_0_5_output in {"("}:
            return position == 5
        elif attn_0_5_output in {")"}:
            return position == 9
        elif attn_0_5_output in {"<s>"}:
            return position == 2

    attn_2_3_pattern = select_closest(positions, attn_0_5_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_6_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_4_output, position):
        if attn_0_4_output in {"("}:
            return position == 4
        elif attn_0_4_output in {")"}:
            return position == 3
        elif attn_0_4_output in {"<s>"}:
            return position == 1

    attn_2_4_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_0_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(token, position):
        if token in {"("}:
            return position == 3
        elif token in {")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 5

    attn_2_5_pattern = select_closest(positions, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(token, position):
        if token in {"("}:
            return position == 2
        elif token in {")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 5

    attn_2_6_pattern = select_closest(positions, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_3_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_5_output, position):
        if attn_0_5_output in {"(", "<s>"}:
            return position == 3
        elif attn_0_5_output in {")"}:
            return position == 9

    attn_2_7_pattern = select_closest(positions, attn_0_5_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_1_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_1_output, attn_1_0_output):
        if attn_1_1_output in {"(", ")", "<s>"}:
            return attn_1_0_output == ""

    num_attn_2_0_pattern = select(attn_1_0_outputs, attn_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_0_output, attn_0_4_output):
        if attn_1_0_output in {"(", ")", "<s>"}:
            return attn_0_4_output == ""

    num_attn_2_1_pattern = select(attn_0_4_outputs, attn_1_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_6_output, mlp_1_0_output):
        if attn_1_6_output in {"(", "<s>"}:
            return mlp_1_0_output == 2
        elif attn_1_6_output in {")"}:
            return mlp_1_0_output == 5

    num_attn_2_2_pattern = select(mlp_1_0_outputs, attn_1_6_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_0_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_3_output, attn_1_6_output):
        if attn_1_3_output in {"(", ")", "<s>"}:
            return attn_1_6_output == ""

    num_attn_2_3_pattern = select(attn_1_6_outputs, attn_1_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_6_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(mlp_0_1_output, mlp_0_0_output):
        if mlp_0_1_output in {0, 32, 34, 5, 41, 42, 75, 44, 15, 49, 51, 55, 24}:
            return mlp_0_0_output == 13
        elif mlp_0_1_output in {1, 67, 4, 46, 19, 25}:
            return mlp_0_0_output == 31
        elif mlp_0_1_output in {2, 9, 10, 43, 47}:
            return mlp_0_0_output == 61
        elif mlp_0_1_output in {33, 3, 37, 40, 79, 16, 18, 21, 23, 58, 59, 29}:
            return mlp_0_0_output == 35
        elif mlp_0_1_output in {6}:
            return mlp_0_0_output == 46
        elif mlp_0_1_output in {7}:
            return mlp_0_0_output == 10
        elif mlp_0_1_output in {8, 12}:
            return mlp_0_0_output == 67
        elif mlp_0_1_output in {65, 35, 11, 71}:
            return mlp_0_0_output == 32
        elif mlp_0_1_output in {13, 52, 27, 28, 61}:
            return mlp_0_0_output == 73
        elif mlp_0_1_output in {66, 68, 14, 26, 30}:
            return mlp_0_0_output == 1
        elif mlp_0_1_output in {36, 38, 72, 73, 45, 78, 48, 17, 22, 56, 57, 62}:
            return mlp_0_0_output == 77
        elif mlp_0_1_output in {20}:
            return mlp_0_0_output == 34
        elif mlp_0_1_output in {31}:
            return mlp_0_0_output == 12
        elif mlp_0_1_output in {39}:
            return mlp_0_0_output == 27
        elif mlp_0_1_output in {50}:
            return mlp_0_0_output == 72
        elif mlp_0_1_output in {53}:
            return mlp_0_0_output == 43
        elif mlp_0_1_output in {64, 54}:
            return mlp_0_0_output == 16
        elif mlp_0_1_output in {60}:
            return mlp_0_0_output == 63
        elif mlp_0_1_output in {63}:
            return mlp_0_0_output == 54
        elif mlp_0_1_output in {69}:
            return mlp_0_0_output == 62
        elif mlp_0_1_output in {70}:
            return mlp_0_0_output == 58
        elif mlp_0_1_output in {74}:
            return mlp_0_0_output == 15
        elif mlp_0_1_output in {76}:
            return mlp_0_0_output == 65
        elif mlp_0_1_output in {77}:
            return mlp_0_0_output == 60

    num_attn_2_4_pattern = select(mlp_0_0_outputs, mlp_0_1_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_4_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(mlp_1_1_output, attn_1_7_output):
        if mlp_1_1_output in {
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
            return attn_1_7_output == ""
        elif mlp_1_1_output in {57}:
            return attn_1_7_output == "("

    num_attn_2_5_pattern = select(attn_1_7_outputs, mlp_1_1_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_2_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_0_output, attn_1_7_output):
        if attn_1_0_output in {"(", ")", "<s>"}:
            return attn_1_7_output == ""

    num_attn_2_6_pattern = select(attn_1_7_outputs, attn_1_0_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_2_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(mlp_1_0_output, mlp_0_0_output):
        if mlp_1_0_output in {
            0,
            2,
            4,
            5,
            6,
            9,
            10,
            12,
            14,
            16,
            17,
            18,
            20,
            22,
            26,
            30,
            31,
            39,
            40,
            41,
            44,
            48,
            53,
            54,
            56,
            57,
            58,
            59,
            63,
            65,
            69,
            70,
            77,
        }:
            return mlp_0_0_output == 28
        elif mlp_1_0_output in {1, 73, 43, 15, 19}:
            return mlp_0_0_output == 31
        elif mlp_1_0_output in {64, 3, 67, 37, 7, 76, 13, 45, 52, 21, 24, 27}:
            return mlp_0_0_output == 32
        elif mlp_1_0_output in {8, 47}:
            return mlp_0_0_output == 61
        elif mlp_1_0_output in {71, 11, 60, 78, 79, 28, 62}:
            return mlp_0_0_output == 77
        elif mlp_1_0_output in {23}:
            return mlp_0_0_output == 14
        elif mlp_1_0_output in {25}:
            return mlp_0_0_output == 15
        elif mlp_1_0_output in {29}:
            return mlp_0_0_output == 67
        elif mlp_1_0_output in {32}:
            return mlp_0_0_output == 64
        elif mlp_1_0_output in {33}:
            return mlp_0_0_output == 49
        elif mlp_1_0_output in {34}:
            return mlp_0_0_output == 24
        elif mlp_1_0_output in {66, 35, 74, 46, 61}:
            return mlp_0_0_output == 13
        elif mlp_1_0_output in {36}:
            return mlp_0_0_output == 22
        elif mlp_1_0_output in {38}:
            return mlp_0_0_output == 69
        elif mlp_1_0_output in {42}:
            return mlp_0_0_output == 70
        elif mlp_1_0_output in {49}:
            return mlp_0_0_output == 43
        elif mlp_1_0_output in {50}:
            return mlp_0_0_output == 42
        elif mlp_1_0_output in {51}:
            return mlp_0_0_output == 76
        elif mlp_1_0_output in {55}:
            return mlp_0_0_output == 47
        elif mlp_1_0_output in {68}:
            return mlp_0_0_output == 63
        elif mlp_1_0_output in {72}:
            return mlp_0_0_output == 46
        elif mlp_1_0_output in {75}:
            return mlp_0_0_output == 57

    num_attn_2_7_pattern = select(mlp_0_0_outputs, mlp_1_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_4_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, attn_2_5_output):
        key = (attn_2_3_output, attn_2_5_output)
        if key in {(")", ")")}:
            return 74
        elif key in {("(", ")"), ("<s>", ")")}:
            return 4
        return 53

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_5_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_2_output, attn_2_4_output):
        key = (attn_2_2_output, attn_2_4_output)
        if key in {("(", "("), ("(", "<s>"), (")", "("), ("<s>", "(")}:
            return 54
        elif key in {(")", ")"), (")", "<s>"), ("<s>", ")"), ("<s>", "<s>")}:
            return 41
        return 46

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_4_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output, num_attn_1_1_output):
        key = (num_attn_1_0_output, num_attn_1_1_output)
        return 36

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_0_output, num_attn_2_7_output):
        key = (num_attn_2_0_output, num_attn_2_7_output)
        return 51

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_2_7_outputs)
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
            ")",
            "(",
            ")",
            "(",
            "(",
            ")",
            "(",
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            ")",
            ")",
            ")",
            ")",
            "(",
            ")",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
        ]
    )
)
