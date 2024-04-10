import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/sort/trainlength40/s4/sort_weights.csv",
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
            return k_position == 21
        elif q_position in {1, 3, 4, 7, 9, 11, 12, 13, 15, 21, 26}:
            return k_position == 5
        elif q_position in {2, 5, 39}:
            return k_position == 3
        elif q_position in {40, 6}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 24
        elif q_position in {10}:
            return k_position == 20
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {16, 25, 32, 33}:
            return k_position == 19
        elif q_position in {17, 18, 22}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 10
        elif q_position in {20}:
            return k_position == 9
        elif q_position in {36, 23}:
            return k_position == 15
        elif q_position in {24}:
            return k_position == 32
        elif q_position in {27}:
            return k_position == 33
        elif q_position in {28, 46}:
            return k_position == 36
        elif q_position in {43, 29, 47}:
            return k_position == 35
        elif q_position in {34, 30}:
            return k_position == 26
        elif q_position in {31}:
            return k_position == 37
        elif q_position in {41, 35}:
            return k_position == 38
        elif q_position in {37}:
            return k_position == 39
        elif q_position in {38}:
            return k_position == 28
        elif q_position in {42, 45}:
            return k_position == 48
        elif q_position in {44}:
            return k_position == 27
        elif q_position in {48}:
            return k_position == 44
        elif q_position in {49}:
            return k_position == 25

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 36, 37, 39, 11, 22, 26, 27, 29}:
            return token == "3"
        elif position in {1, 2, 3, 4, 5, 6, 7, 8, 38, 30}:
            return token == "2"
        elif position in {32, 34, 9, 12, 13, 14, 15, 17, 18, 19, 20, 23, 25, 28}:
            return token == "4"
        elif position in {10}:
            return token == "0"
        elif position in {16, 24, 21, 31}:
            return token == "1"
        elif position in {33}:
            return token == "<s>"
        elif position in {35}:
            return token == "</s>"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 1, 3, 37, 38, 10, 27, 29}:
            return token == "2"
        elif position in {33, 2, 36, 12, 17, 22, 24, 26, 28, 30}:
            return token == "3"
        elif position in {32, 34, 4, 6, 7, 8, 9, 11, 43, 19, 20, 25}:
            return token == "4"
        elif position in {5, 40, 41, 42, 44, 45, 14, 46, 16, 47, 48, 49, 21, 23}:
            return token == ""
        elif position in {13}:
            return token == "0"
        elif position in {18, 31, 39, 15}:
            return token == "1"
        elif position in {35}:
            return token == "<s>"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {
            0,
            1,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            18,
            20,
            22,
            25,
            26,
            28,
            30,
            31,
            32,
            34,
            36,
        }:
            return token == "3"
        elif position in {2}:
            return token == "0"
        elif position in {17, 12, 23}:
            return token == "4"
        elif position in {24, 29, 19, 21}:
            return token == "1"
        elif position in {33, 35, 37, 38, 39, 27}:
            return token == "2"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 2, 4, 9, 28, 31}:
            return token == "3"
        elif position in {32, 1, 14, 19, 24, 29}:
            return token == "1"
        elif position in {3, 6, 11, 12, 15, 16, 17, 18, 21, 22, 25, 26, 30}:
            return token == "4"
        elif position in {35, 36, 5, 39, 10, 13, 20, 23}:
            return token == "2"
        elif position in {37, 38, 7, 8, 27}:
            return token == "0"
        elif position in {33}:
            return token == "<pad>"
        elif position in {34}:
            return token == "</s>"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 1, 2, 35, 4, 36, 38, 39, 9, 11, 20, 21, 23, 25, 26, 29, 31}:
            return token == "3"
        elif position in {3, 6, 7, 12, 14}:
            return token == "2"
        elif position in {32, 34, 5, 8, 10, 13, 16, 19, 27}:
            return token == "4"
        elif position in {17, 18, 22, 15}:
            return token == "1"
        elif position in {33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 24, 28, 30}:
            return token == ""
        elif position in {37}:
            return token == "<pad>"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 15}:
            return k_position == 9
        elif q_position in {1, 30}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 20
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4, 6}:
            return k_position == 12
        elif q_position in {36, 5}:
            return k_position == 13
        elif q_position in {7}:
            return k_position == 16
        elif q_position in {8, 10}:
            return k_position == 14
        elif q_position in {9}:
            return k_position == 18
        elif q_position in {11, 12, 13}:
            return k_position == 15
        elif q_position in {32, 14}:
            return k_position == 26
        elif q_position in {16, 17}:
            return k_position == 22
        elif q_position in {18, 22}:
            return k_position == 29
        elif q_position in {19}:
            return k_position == 21
        elif q_position in {20}:
            return k_position == 17
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {24, 38, 23}:
            return k_position == 30
        elif q_position in {25}:
            return k_position == 3
        elif q_position in {26, 37}:
            return k_position == 6
        elif q_position in {27, 29}:
            return k_position == 33
        elif q_position in {28}:
            return k_position == 36
        elif q_position in {31}:
            return k_position == 24
        elif q_position in {33}:
            return k_position == 7
        elif q_position in {34}:
            return k_position == 2
        elif q_position in {35, 45}:
            return k_position == 27
        elif q_position in {46, 39}:
            return k_position == 34
        elif q_position in {40}:
            return k_position == 40
        elif q_position in {41, 42}:
            return k_position == 35
        elif q_position in {48, 43}:
            return k_position == 31
        elif q_position in {44}:
            return k_position == 38
        elif q_position in {47}:
            return k_position == 42
        elif q_position in {49}:
            return k_position == 45

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, positions)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1, 40, 9, 13, 47}:
            return k_position == 1
        elif q_position in {2, 43, 45, 14, 48, 17, 27}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {49, 6}:
            return k_position == 2
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {10, 36, 44}:
            return k_position == 3
        elif q_position in {11, 28}:
            return k_position == 26
        elif q_position in {12, 39}:
            return k_position == 14
        elif q_position in {29, 15}:
            return k_position == 17
        elif q_position in {16}:
            return k_position == 6
        elif q_position in {18, 22}:
            return k_position == 21
        elif q_position in {32, 41, 19}:
            return k_position == 30
        elif q_position in {20}:
            return k_position == 27
        elif q_position in {21}:
            return k_position == 19
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 31
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {37, 30}:
            return k_position == 28
        elif q_position in {31}:
            return k_position == 39
        elif q_position in {33}:
            return k_position == 29
        elif q_position in {34}:
            return k_position == 33
        elif q_position in {35}:
            return k_position == 32
        elif q_position in {38}:
            return k_position == 0
        elif q_position in {42, 46}:
            return k_position == 13

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 16}:
            return k_position == 19
        elif q_position in {40, 1, 47}:
            return k_position == 4
        elif q_position in {2, 4}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {41, 6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8, 39}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {48, 10}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 16
        elif q_position in {13, 14}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {17, 46}:
            return k_position == 20
        elif q_position in {18}:
            return k_position == 22
        elif q_position in {19, 20}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 25
        elif q_position in {22}:
            return k_position == 26
        elif q_position in {24, 43, 45, 23}:
            return k_position == 28
        elif q_position in {25, 26}:
            return k_position == 30
        elif q_position in {27}:
            return k_position == 32
        elif q_position in {28}:
            return k_position == 33
        elif q_position in {29}:
            return k_position == 34
        elif q_position in {30}:
            return k_position == 35
        elif q_position in {31}:
            return k_position == 36
        elif q_position in {32}:
            return k_position == 37
        elif q_position in {33, 42}:
            return k_position == 43
        elif q_position in {34}:
            return k_position == 38
        elif q_position in {35}:
            return k_position == 45
        elif q_position in {36}:
            return k_position == 41
        elif q_position in {44, 37}:
            return k_position == 46
        elif q_position in {38}:
            return k_position == 49
        elif q_position in {49}:
            return k_position == 21

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {
            0,
            1,
            2,
            3,
            4,
            6,
            7,
            9,
            11,
            12,
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
        }:
            return token == ""
        elif position in {13, 18, 5, 15}:
            return token == "</s>"
        elif position in {8}:
            return token == "<pad>"
        elif position in {39, 10, 14, 16, 17, 19}:
            return token == "<s>"
        elif position in {33, 35, 20, 21, 24, 31}:
            return token == "1"
        elif position in {32, 34, 36, 37, 38, 22, 23, 25, 26, 27, 28, 29, 30}:
            return token == "0"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {
            0,
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
        }:
            return token == ""
        elif position in {1, 11}:
            return token == "</s>"
        elif position in {2, 3, 4, 5, 6, 7, 8, 9, 10, 39}:
            return token == "0"
        elif position in {38}:
            return token == "<pad>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 12, 13, 14}:
            return k_position == 19
        elif q_position in {48, 1, 31}:
            return k_position == 46
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 16
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {15}:
            return k_position == 22
        elif q_position in {16, 42}:
            return k_position == 26
        elif q_position in {17, 18, 45, 47}:
            return k_position == 25
        elif q_position in {19}:
            return k_position == 27
        elif q_position in {20}:
            return k_position == 28
        elif q_position in {43, 21, 22, 23, 24}:
            return k_position == 32
        elif q_position in {25, 46}:
            return k_position == 33
        elif q_position in {26}:
            return k_position == 34
        elif q_position in {27, 28}:
            return k_position == 35
        elif q_position in {29}:
            return k_position == 37
        elif q_position in {30}:
            return k_position == 38
        elif q_position in {32}:
            return k_position == 48
        elif q_position in {33, 34}:
            return k_position == 40
        elif q_position in {35}:
            return k_position == 47
        elif q_position in {36, 37, 38}:
            return k_position == 44
        elif q_position in {39}:
            return k_position == 0
        elif q_position in {40, 49}:
            return k_position == 42
        elif q_position in {41}:
            return k_position == 29
        elif q_position in {44}:
            return k_position == 43

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 2, 3, 4, 5, 6, 7}:
            return token == "1"
        elif position in {1}:
            return token == "</s>"
        elif position in {
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
        }:
            return token == ""
        elif position in {39}:
            return token == "0"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {
            0,
            5,
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
            34,
            35,
            36,
            37,
            38,
        }:
            return token == ""
        elif position in {1, 19, 20, 17}:
            return token == "</s>"
        elif position in {2, 4, 6, 7, 46}:
            return token == "2"
        elif position in {3}:
            return token == "<s>"
        elif position in {8, 9, 10, 11, 12, 40, 41, 42, 43, 44, 45, 47, 48, 49}:
            return token == "1"
        elif position in {39, 13, 14, 15, 16, 18}:
            return token == "0"
        elif position in {33}:
            return token == "<pad>"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 36, 37, 39}:
            return token == "2"
        elif position in {32, 1, 34}:
            return token == "0"
        elif position in {
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
            14,
            15,
            16,
            17,
            18,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif position in {13}:
            return token == "<pad>"
        elif position in {24, 19, 45, 23}:
            return token == "</s>"
        elif position in {20, 21, 22, 25, 26}:
            return token == "<s>"
        elif position in {27, 28, 29, 30, 31}:
            return token == "4"
        elif position in {33, 35, 38}:
            return token == "1"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 1, 2, 3, 4, 39}:
            return token == "0"
        elif position in {
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
        }:
            return token == ""
        elif position in {21}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {("0", 19)}:
            return 33
        return 36

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(token, attn_0_4_output):
        key = (token, attn_0_4_output)
        return 1

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(tokens, attn_0_4_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(position):
        key = position
        if key in {9, 10, 16, 17, 18, 19, 20}:
            return 27
        elif key in {1, 2, 3, 4, 5, 39}:
            return 31
        elif key in {0, 6, 7, 8, 21, 22}:
            return 49
        return 13

    mlp_0_2_outputs = [mlp_0_2(k0) for k0 in positions]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position):
        key = position
        if key in {3, 11, 14, 15, 16, 17, 23, 27, 28, 31}:
            return 27
        elif key in {13, 20, 21, 22, 24, 32}:
            return 15
        elif key in {34, 35, 36, 37}:
            return 25
        elif key in {0, 1, 2}:
            return 20
        return 19

    mlp_0_3_outputs = [mlp_0_3(k0) for k0 in positions]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_4_output):
        key = (num_attn_0_0_output, num_attn_0_4_output)
        return 2

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_4_output):
        key = (num_attn_0_3_output, num_attn_0_4_output)
        return 28

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 48

    num_mlp_0_2_outputs = [num_mlp_0_2(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_6_output, num_attn_0_7_output):
        key = (num_attn_0_6_output, num_attn_0_7_output)
        return 33

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, attn_0_2_output):
        if position in {0}:
            return attn_0_2_output == "</s>"
        elif position in {
            1,
            3,
            4,
            5,
            8,
            11,
            13,
            14,
            15,
            16,
            18,
            19,
            21,
            22,
            23,
            24,
            26,
            27,
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
            45,
            48,
            49,
        }:
            return attn_0_2_output == ""
        elif position in {2, 29, 6}:
            return attn_0_2_output == "<s>"
        elif position in {10, 7}:
            return attn_0_2_output == "<pad>"
        elif position in {9, 25}:
            return attn_0_2_output == "4"
        elif position in {12, 44, 47}:
            return attn_0_2_output == "2"
        elif position in {17, 28, 46, 31}:
            return attn_0_2_output == "0"
        elif position in {20}:
            return attn_0_2_output == "1"
        elif position in {30}:
            return attn_0_2_output == "3"

    attn_1_0_pattern = select_closest(attn_0_2_outputs, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 34, 35, 8, 41, 16, 21, 31}:
            return token == "2"
        elif mlp_0_0_output in {1, 3, 6, 39, 40, 9, 13, 46, 47, 49, 19, 22, 30}:
            return token == ""
        elif mlp_0_0_output in {2, 4, 36, 7, 10, 14, 18, 20, 23, 25, 26, 27}:
            return token == "4"
        elif mlp_0_0_output in {42, 12, 5}:
            return token == "<s>"
        elif mlp_0_0_output in {33, 11, 45, 17, 24, 29}:
            return token == "3"
        elif mlp_0_0_output in {43, 28, 44, 15}:
            return token == "1"
        elif mlp_0_0_output in {32, 48, 37, 38}:
            return token == "</s>"

    attn_1_1_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_3_output, position):
        if mlp_0_3_output in {0}:
            return position == 15
        elif mlp_0_3_output in {1, 13, 5}:
            return position == 18
        elif mlp_0_3_output in {2}:
            return position == 19
        elif mlp_0_3_output in {42, 3, 39}:
            return position == 39
        elif mlp_0_3_output in {4}:
            return position == 37
        elif mlp_0_3_output in {33, 36, 6, 7}:
            return position == 0
        elif mlp_0_3_output in {8, 43}:
            return position == 10
        elif mlp_0_3_output in {16, 9, 22, 15}:
            return position == 2
        elif mlp_0_3_output in {10}:
            return position == 16
        elif mlp_0_3_output in {32, 11, 46, 47, 26, 27}:
            return position == 4
        elif mlp_0_3_output in {12}:
            return position == 23
        elif mlp_0_3_output in {14}:
            return position == 17
        elif mlp_0_3_output in {17, 19, 30}:
            return position == 6
        elif mlp_0_3_output in {18}:
            return position == 3
        elif mlp_0_3_output in {20}:
            return position == 1
        elif mlp_0_3_output in {21}:
            return position == 12
        elif mlp_0_3_output in {48, 23}:
            return position == 5
        elif mlp_0_3_output in {24, 34, 31}:
            return position == 9
        elif mlp_0_3_output in {25}:
            return position == 13
        elif mlp_0_3_output in {28}:
            return position == 11
        elif mlp_0_3_output in {40, 29, 38}:
            return position == 7
        elif mlp_0_3_output in {35}:
            return position == 32
        elif mlp_0_3_output in {37}:
            return position == 8
        elif mlp_0_3_output in {41}:
            return position == 38
        elif mlp_0_3_output in {44}:
            return position == 33
        elif mlp_0_3_output in {45}:
            return position == 26
        elif mlp_0_3_output in {49}:
            return position == 22

    attn_1_2_pattern = select_closest(positions, mlp_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_7_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, num_mlp_0_1_output):
        if position in {0}:
            return num_mlp_0_1_output == 28
        elif position in {1, 49}:
            return num_mlp_0_1_output == 1
        elif position in {2}:
            return num_mlp_0_1_output == 15
        elif position in {3}:
            return num_mlp_0_1_output == 26
        elif position in {17, 4}:
            return num_mlp_0_1_output == 18
        elif position in {35, 5, 6, 7, 12, 16, 30}:
            return num_mlp_0_1_output == 27
        elif position in {8, 19, 11}:
            return num_mlp_0_1_output == 20
        elif position in {9, 34, 13, 15}:
            return num_mlp_0_1_output == 31
        elif position in {10}:
            return num_mlp_0_1_output == 23
        elif position in {20, 21, 14}:
            return num_mlp_0_1_output == 16
        elif position in {18}:
            return num_mlp_0_1_output == 5
        elif position in {22}:
            return num_mlp_0_1_output == 36
        elif position in {23}:
            return num_mlp_0_1_output == 22
        elif position in {24}:
            return num_mlp_0_1_output == 17
        elif position in {25}:
            return num_mlp_0_1_output == 19
        elif position in {26}:
            return num_mlp_0_1_output == 10
        elif position in {36, 37, 38, 40, 41, 47, 27, 28}:
            return num_mlp_0_1_output == 6
        elif position in {29}:
            return num_mlp_0_1_output == 30
        elif position in {31}:
            return num_mlp_0_1_output == 35
        elif position in {32}:
            return num_mlp_0_1_output == 4
        elif position in {33}:
            return num_mlp_0_1_output == 7
        elif position in {39}:
            return num_mlp_0_1_output == 3
        elif position in {42}:
            return num_mlp_0_1_output == 49
        elif position in {43, 45}:
            return num_mlp_0_1_output == 0
        elif position in {44, 46}:
            return num_mlp_0_1_output == 33
        elif position in {48}:
            return num_mlp_0_1_output == 9

    attn_1_3_pattern = select_closest(num_mlp_0_1_outputs, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_5_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_6_output, token):
        if attn_0_6_output in {0, 6}:
            return token == "<s>"
        elif attn_0_6_output in {1, 7, 8, 9, 12, 13, 16, 17, 21, 23, 24, 26, 28}:
            return token == "4"
        elif attn_0_6_output in {2, 5, 40, 14, 47, 49}:
            return token == ""
        elif attn_0_6_output in {32, 34, 3, 37, 11}:
            return token == "3"
        elif attn_0_6_output in {35, 4}:
            return token == "<pad>"
        elif attn_0_6_output in {33, 36, 38, 39, 10, 15, 18, 19, 29, 31}:
            return token == "2"
        elif attn_0_6_output in {25, 27, 20, 30}:
            return token == "0"
        elif attn_0_6_output in {22}:
            return token == "1"
        elif attn_0_6_output in {41, 42, 43, 44, 45, 46, 48}:
            return token == "</s>"

    attn_1_4_pattern = select_closest(tokens, attn_0_6_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_5_output, token):
        if attn_0_5_output in {"0"}:
            return token == "1"
        elif attn_0_5_output in {"</s>", "3", "1"}:
            return token == "4"
        elif attn_0_5_output in {"2"}:
            return token == "3"
        elif attn_0_5_output in {"4"}:
            return token == "2"
        elif attn_0_5_output in {"<s>"}:
            return token == "<s>"

    attn_1_5_pattern = select_closest(tokens, attn_0_5_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_5_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, token):
        if position in {
            0,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            19,
            23,
            25,
            26,
            29,
            32,
            34,
            35,
            37,
            46,
        }:
            return token == "4"
        elif position in {1, 2, 3, 39, 41, 17, 18, 20}:
            return token == "1"
        elif position in {40, 4, 45}:
            return token == "<s>"
        elif position in {5, 38, 12, 21, 24}:
            return token == "2"
        elif position in {33, 36, 22, 30}:
            return token == "3"
        elif position in {43, 44, 47, 48, 49, 27, 31}:
            return token == ""
        elif position in {28}:
            return token == "</s>"
        elif position in {42}:
            return token == "0"

    attn_1_6_pattern = select_closest(tokens, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_position, k_position):
        if q_position in {0, 3, 38, 39}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {41, 2}:
            return k_position == 1
        elif q_position in {40, 4, 47}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {6, 7}:
            return k_position == 9
        elif q_position in {8, 11}:
            return k_position == 12
        elif q_position in {9, 12}:
            return k_position == 13
        elif q_position in {33, 10}:
            return k_position == 5
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {49, 14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 20
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 21
        elif q_position in {20}:
            return k_position == 22
        elif q_position in {21, 23}:
            return k_position == 25
        elif q_position in {22}:
            return k_position == 23
        elif q_position in {24, 25}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 27
        elif q_position in {27}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 30
        elif q_position in {29}:
            return k_position == 34
        elif q_position in {30}:
            return k_position == 35
        elif q_position in {31}:
            return k_position == 36
        elif q_position in {32}:
            return k_position == 37
        elif q_position in {34}:
            return k_position == 10
        elif q_position in {35}:
            return k_position == 29
        elif q_position in {36}:
            return k_position == 6
        elif q_position in {37}:
            return k_position == 16
        elif q_position in {48, 42, 45}:
            return k_position == 0
        elif q_position in {43}:
            return k_position == 45
        elif q_position in {44}:
            return k_position == 33
        elif q_position in {46}:
            return k_position == 24

    attn_1_7_pattern = select_closest(positions, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 34}:
            return k_num_mlp_0_0_output == 20
        elif q_num_mlp_0_0_output in {1}:
            return k_num_mlp_0_0_output == 27
        elif q_num_mlp_0_0_output in {2, 27}:
            return k_num_mlp_0_0_output == 46
        elif q_num_mlp_0_0_output in {19, 3}:
            return k_num_mlp_0_0_output == 7
        elif q_num_mlp_0_0_output in {4}:
            return k_num_mlp_0_0_output == 19
        elif q_num_mlp_0_0_output in {24, 5}:
            return k_num_mlp_0_0_output == 25
        elif q_num_mlp_0_0_output in {35, 6}:
            return k_num_mlp_0_0_output == 39
        elif q_num_mlp_0_0_output in {48, 11, 22, 7}:
            return k_num_mlp_0_0_output == 3
        elif q_num_mlp_0_0_output in {8, 17}:
            return k_num_mlp_0_0_output == 45
        elif q_num_mlp_0_0_output in {9}:
            return k_num_mlp_0_0_output == 35
        elif q_num_mlp_0_0_output in {10, 44}:
            return k_num_mlp_0_0_output == 17
        elif q_num_mlp_0_0_output in {12}:
            return k_num_mlp_0_0_output == 41
        elif q_num_mlp_0_0_output in {13, 30}:
            return k_num_mlp_0_0_output == 26
        elif q_num_mlp_0_0_output in {14}:
            return k_num_mlp_0_0_output == 37
        elif q_num_mlp_0_0_output in {15}:
            return k_num_mlp_0_0_output == 18
        elif q_num_mlp_0_0_output in {16}:
            return k_num_mlp_0_0_output == 24
        elif q_num_mlp_0_0_output in {18}:
            return k_num_mlp_0_0_output == 33
        elif q_num_mlp_0_0_output in {32, 33, 40, 47, 20, 31}:
            return k_num_mlp_0_0_output == 21
        elif q_num_mlp_0_0_output in {21}:
            return k_num_mlp_0_0_output == 2
        elif q_num_mlp_0_0_output in {37, 38, 23}:
            return k_num_mlp_0_0_output == 9
        elif q_num_mlp_0_0_output in {25}:
            return k_num_mlp_0_0_output == 49
        elif q_num_mlp_0_0_output in {26}:
            return k_num_mlp_0_0_output == 14
        elif q_num_mlp_0_0_output in {28}:
            return k_num_mlp_0_0_output == 43
        elif q_num_mlp_0_0_output in {29}:
            return k_num_mlp_0_0_output == 30
        elif q_num_mlp_0_0_output in {36}:
            return k_num_mlp_0_0_output == 28
        elif q_num_mlp_0_0_output in {39}:
            return k_num_mlp_0_0_output == 34
        elif q_num_mlp_0_0_output in {41, 43}:
            return k_num_mlp_0_0_output == 47
        elif q_num_mlp_0_0_output in {42}:
            return k_num_mlp_0_0_output == 36
        elif q_num_mlp_0_0_output in {45}:
            return k_num_mlp_0_0_output == 29
        elif q_num_mlp_0_0_output in {46}:
            return k_num_mlp_0_0_output == 11
        elif q_num_mlp_0_0_output in {49}:
            return k_num_mlp_0_0_output == 6

    num_attn_1_0_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_0
    )
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_5_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_0_output, attn_0_6_output):
        if num_mlp_0_0_output in {0, 32, 34, 40, 41, 43, 49, 22, 28, 30}:
            return attn_0_6_output == 19
        elif num_mlp_0_0_output in {1, 36}:
            return attn_0_6_output == 21
        elif num_mlp_0_0_output in {33, 2, 35, 11, 44, 13, 48, 20, 25}:
            return attn_0_6_output == 9
        elif num_mlp_0_0_output in {3}:
            return attn_0_6_output == 16
        elif num_mlp_0_0_output in {4}:
            return attn_0_6_output == 6
        elif num_mlp_0_0_output in {16, 18, 5, 14}:
            return attn_0_6_output == 20
        elif num_mlp_0_0_output in {6}:
            return attn_0_6_output == 10
        elif num_mlp_0_0_output in {7, 9, 10, 12, 45, 46, 17, 19, 23, 27}:
            return attn_0_6_output == 13
        elif num_mlp_0_0_output in {8, 47}:
            return attn_0_6_output == 29
        elif num_mlp_0_0_output in {15}:
            return attn_0_6_output == 3
        elif num_mlp_0_0_output in {37, 38, 39, 21, 24}:
            return attn_0_6_output == 30
        elif num_mlp_0_0_output in {26}:
            return attn_0_6_output == 23
        elif num_mlp_0_0_output in {29, 31}:
            return attn_0_6_output == 34
        elif num_mlp_0_0_output in {42}:
            return attn_0_6_output == 22

    num_attn_1_1_pattern = select(
        attn_0_6_outputs, num_mlp_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_6_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, attn_0_1_output):
        if position in {
            0,
            4,
            5,
            6,
            7,
            18,
            19,
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
            43,
        }:
            return attn_0_1_output == ""
        elif position in {1, 3}:
            return attn_0_1_output == "</s>"
        elif position in {2}:
            return attn_0_1_output == "2"
        elif position in {
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            40,
            41,
            42,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return attn_0_1_output == "1"
        elif position in {17, 20}:
            return attn_0_1_output == "<s>"

    num_attn_1_2_pattern = select(attn_0_1_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_7_output):
        if position in {
            0,
            4,
            10,
            15,
            16,
            18,
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
            46,
            47,
            48,
        }:
            return attn_0_7_output == ""
        elif position in {1, 3, 9, 14, 20, 21}:
            return attn_0_7_output == "</s>"
        elif position in {17, 2, 19}:
            return attn_0_7_output == "<s>"
        elif position in {5, 6, 7, 41, 11, 12, 43, 45, 49}:
            return attn_0_7_output == "0"
        elif position in {8, 44}:
            return attn_0_7_output == "2"
        elif position in {42, 13}:
            return attn_0_7_output == "1"

    num_attn_1_3_pattern = select(attn_0_7_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, attn_0_4_output):
        if position in {0, 1, 2, 8, 14, 15, 18, 19, 22}:
            return attn_0_4_output == "1"
        elif position in {16, 3, 31}:
            return attn_0_4_output == "</s>"
        elif position in {
            4,
            9,
            10,
            13,
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
            35,
            36,
            37,
            38,
            47,
        }:
            return attn_0_4_output == ""
        elif position in {
            5,
            6,
            7,
            39,
            40,
            41,
            11,
            12,
            42,
            43,
            44,
            45,
            17,
            46,
            48,
            20,
            21,
        }:
            return attn_0_4_output == "0"
        elif position in {49}:
            return attn_0_4_output == "<s>"

    num_attn_1_4_pattern = select(attn_0_4_outputs, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_2_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, token):
        if position in {
            0,
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
        }:
            return token == ""
        elif position in {1, 2, 4, 8, 9, 10, 40, 44, 13, 49}:
            return token == "1"
        elif position in {3, 5}:
            return token == "2"
        elif position in {6, 7, 41, 42, 11, 12, 43, 45, 46, 47, 48}:
            return token == "0"
        elif position in {14}:
            return token == "<s>"

    num_attn_1_5_pattern = select(tokens, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_5_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, attn_0_6_output):
        if position in {0, 42}:
            return attn_0_6_output == 44
        elif position in {1, 2, 3, 4}:
            return attn_0_6_output == 7
        elif position in {5}:
            return attn_0_6_output == 8
        elif position in {6, 40, 41, 43, 45, 47}:
            return attn_0_6_output == 1
        elif position in {9, 7}:
            return attn_0_6_output == 12
        elif position in {8}:
            return attn_0_6_output == 16
        elif position in {10}:
            return attn_0_6_output == 20
        elif position in {11}:
            return attn_0_6_output == 21
        elif position in {12}:
            return attn_0_6_output == 17
        elif position in {13, 14}:
            return attn_0_6_output == 23
        elif position in {15}:
            return attn_0_6_output == 25
        elif position in {16, 44}:
            return attn_0_6_output == 33
        elif position in {17, 21, 22, 49}:
            return attn_0_6_output == 29
        elif position in {18}:
            return attn_0_6_output == 24
        elif position in {19, 20}:
            return attn_0_6_output == 26
        elif position in {23}:
            return attn_0_6_output == 36
        elif position in {24, 25}:
            return attn_0_6_output == 32
        elif position in {36, 26, 28, 29, 30}:
            return attn_0_6_output == 49
        elif position in {27}:
            return attn_0_6_output == 34
        elif position in {32, 35, 46, 31}:
            return attn_0_6_output == 47
        elif position in {33}:
            return attn_0_6_output == 40
        elif position in {34}:
            return attn_0_6_output == 45
        elif position in {37}:
            return attn_0_6_output == 46
        elif position in {38}:
            return attn_0_6_output == 42
        elif position in {39}:
            return attn_0_6_output == 48
        elif position in {48}:
            return attn_0_6_output == 27

    num_attn_1_6_pattern = select(attn_0_6_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_5_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {0, 14, 15}:
            return k_attn_0_6_output == 17
        elif q_attn_0_6_output in {40, 1, 41}:
            return k_attn_0_6_output == 11
        elif q_attn_0_6_output in {2, 3, 6}:
            return k_attn_0_6_output == 27
        elif q_attn_0_6_output in {4}:
            return k_attn_0_6_output == 3
        elif q_attn_0_6_output in {8, 10, 5, 7}:
            return k_attn_0_6_output == 9
        elif q_attn_0_6_output in {9, 44, 16, 48, 49}:
            return k_attn_0_6_output == 19
        elif q_attn_0_6_output in {32, 26, 11}:
            return k_attn_0_6_output == 37
        elif q_attn_0_6_output in {12, 28}:
            return k_attn_0_6_output == 16
        elif q_attn_0_6_output in {19, 13}:
            return k_attn_0_6_output == 34
        elif q_attn_0_6_output in {17}:
            return k_attn_0_6_output == 43
        elif q_attn_0_6_output in {18, 42}:
            return k_attn_0_6_output == 4
        elif q_attn_0_6_output in {20}:
            return k_attn_0_6_output == 48
        elif q_attn_0_6_output in {21}:
            return k_attn_0_6_output == 26
        elif q_attn_0_6_output in {22}:
            return k_attn_0_6_output == 25
        elif q_attn_0_6_output in {39, 23}:
            return k_attn_0_6_output == 30
        elif q_attn_0_6_output in {24, 30}:
            return k_attn_0_6_output == 35
        elif q_attn_0_6_output in {25, 36}:
            return k_attn_0_6_output == 14
        elif q_attn_0_6_output in {27, 37}:
            return k_attn_0_6_output == 29
        elif q_attn_0_6_output in {29}:
            return k_attn_0_6_output == 7
        elif q_attn_0_6_output in {31}:
            return k_attn_0_6_output == 10
        elif q_attn_0_6_output in {33, 45}:
            return k_attn_0_6_output == 13
        elif q_attn_0_6_output in {34}:
            return k_attn_0_6_output == 47
        elif q_attn_0_6_output in {35}:
            return k_attn_0_6_output == 15
        elif q_attn_0_6_output in {38}:
            return k_attn_0_6_output == 12
        elif q_attn_0_6_output in {43}:
            return k_attn_0_6_output == 21
        elif q_attn_0_6_output in {46}:
            return k_attn_0_6_output == 8
        elif q_attn_0_6_output in {47}:
            return k_attn_0_6_output == 2

    num_attn_1_7_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_6_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, num_mlp_0_2_output):
        key = (attn_1_2_output, num_mlp_0_2_output)
        return 43

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, num_mlp_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_1_output, attn_1_2_output):
        key = (attn_1_1_output, attn_1_2_output)
        if key in {("0", "4"), ("2", "4"), ("3", "4"), ("4", "4"), ("</s>", "4")}:
            return 46
        return 18

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_1_output, position):
        key = (attn_1_1_output, position)
        if key in {
            ("1", 7),
            ("1", 34),
            ("2", 6),
            ("2", 7),
            ("2", 30),
            ("2", 32),
            ("2", 34),
            ("3", 7),
            ("3", 34),
            ("4", 7),
            ("4", 34),
            ("</s>", 34),
        }:
            return 17
        elif key in {
            ("0", 36),
            ("1", 36),
            ("2", 36),
            ("3", 36),
            ("4", 36),
            ("</s>", 36),
            ("<s>", 36),
        }:
            return 30
        elif key in {("2", 4), ("</s>", 7)}:
            return 26
        return 1

    mlp_1_2_outputs = [mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_1_outputs, positions)]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_7_output, mlp_0_3_output):
        key = (attn_1_7_output, mlp_0_3_output)
        if key in {("1", 21), ("</s>", 21)}:
            return 25
        elif key in {
            ("0", 20),
            ("1", 1),
            ("1", 2),
            ("1", 3),
            ("1", 6),
            ("1", 7),
            ("1", 8),
            ("1", 20),
            ("1", 23),
            ("1", 26),
            ("1", 28),
            ("1", 31),
            ("1", 32),
            ("1", 45),
            ("1", 46),
            ("1", 49),
        }:
            return 27
        return 13

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_7_outputs, mlp_0_3_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_7_output):
        key = num_attn_1_7_output
        return 13

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_7_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_1_0_output):
        key = (num_attn_1_1_output, num_attn_1_0_output)
        return 43

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_0_7_output, num_attn_1_1_output):
        key = (num_attn_0_7_output, num_attn_1_1_output)
        return 22

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_7_output, num_attn_1_2_output):
        key = (num_attn_1_7_output, num_attn_1_2_output)
        return 46

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "1"
        elif q_token in {"4", "1"}:
            return k_token == "0"
        elif q_token in {"<s>", "3"}:
            return k_token == "4"
        elif q_token in {"</s>"}:
            return k_token == "<s>"

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_4_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"3", "2"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"</s>"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == "</s>"

    attn_2_1_pattern = select_closest(tokens, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_3_output, num_mlp_1_0_output):
        if mlp_0_3_output in {0, 32, 9, 42, 48, 19, 21, 22, 26}:
            return num_mlp_1_0_output == 7
        elif mlp_0_3_output in {1}:
            return num_mlp_1_0_output == 27
        elif mlp_0_3_output in {2, 34}:
            return num_mlp_1_0_output == 45
        elif mlp_0_3_output in {3}:
            return num_mlp_1_0_output == 47
        elif mlp_0_3_output in {4, 36}:
            return num_mlp_1_0_output == 1
        elif mlp_0_3_output in {11, 5}:
            return num_mlp_1_0_output == 0
        elif mlp_0_3_output in {10, 6}:
            return num_mlp_1_0_output == 23
        elif mlp_0_3_output in {41, 7}:
            return num_mlp_1_0_output == 6
        elif mlp_0_3_output in {8, 35}:
            return num_mlp_1_0_output == 26
        elif mlp_0_3_output in {12, 45}:
            return num_mlp_1_0_output == 41
        elif mlp_0_3_output in {13}:
            return num_mlp_1_0_output == 24
        elif mlp_0_3_output in {14}:
            return num_mlp_1_0_output == 48
        elif mlp_0_3_output in {15}:
            return num_mlp_1_0_output == 35
        elif mlp_0_3_output in {16, 27}:
            return num_mlp_1_0_output == 18
        elif mlp_0_3_output in {17}:
            return num_mlp_1_0_output == 38
        elif mlp_0_3_output in {18}:
            return num_mlp_1_0_output == 19
        elif mlp_0_3_output in {20, 47}:
            return num_mlp_1_0_output == 2
        elif mlp_0_3_output in {23}:
            return num_mlp_1_0_output == 20
        elif mlp_0_3_output in {24}:
            return num_mlp_1_0_output == 14
        elif mlp_0_3_output in {25, 31}:
            return num_mlp_1_0_output == 5
        elif mlp_0_3_output in {33, 28}:
            return num_mlp_1_0_output == 43
        elif mlp_0_3_output in {29}:
            return num_mlp_1_0_output == 8
        elif mlp_0_3_output in {30}:
            return num_mlp_1_0_output == 44
        elif mlp_0_3_output in {37}:
            return num_mlp_1_0_output == 36
        elif mlp_0_3_output in {38}:
            return num_mlp_1_0_output == 28
        elif mlp_0_3_output in {39}:
            return num_mlp_1_0_output == 39
        elif mlp_0_3_output in {40}:
            return num_mlp_1_0_output == 10
        elif mlp_0_3_output in {43}:
            return num_mlp_1_0_output == 25
        elif mlp_0_3_output in {49, 44}:
            return num_mlp_1_0_output == 40
        elif mlp_0_3_output in {46}:
            return num_mlp_1_0_output == 46

    attn_2_2_pattern = select_closest(
        num_mlp_1_0_outputs, mlp_0_3_outputs, predicate_2_2
    )
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"2", "4", "0", "1"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"</s>", "<s>"}:
            return k_token == "<s>"

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_token, k_token):
        if q_token in {"4", "0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "4"
        elif q_token in {"2"}:
            return k_token == "<pad>"
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"</s>"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == "</s>"

    attn_2_4_pattern = select_closest(tokens, tokens, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_1_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(q_token, k_token):
        if q_token in {"</s>", "0"}:
            return k_token == "<s>"
        elif q_token in {"1"}:
            return k_token == "3"
        elif q_token in {"2"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "2"
        elif q_token in {"<s>", "4"}:
            return k_token == "</s>"

    attn_2_5_pattern = select_closest(tokens, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
        if q_token in {"3", "0", "1"}:
            return k_token == ""
        elif q_token in {"2"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "<pad>"
        elif q_token in {"</s>"}:
            return k_token == "</s>"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_7_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_5_output, position):
        if attn_0_5_output in {"0"}:
            return position == 39
        elif attn_0_5_output in {"4", "1"}:
            return position == 7
        elif attn_0_5_output in {"2"}:
            return position == 4
        elif attn_0_5_output in {"3"}:
            return position == 2
        elif attn_0_5_output in {"</s>"}:
            return position == 13
        elif attn_0_5_output in {"<s>"}:
            return position == 0

    attn_2_7_pattern = select_closest(positions, attn_0_5_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_1_output, attn_1_7_output):
        if attn_1_1_output in {"<s>", "</s>", "0", "2", "4", "1"}:
            return attn_1_7_output == ""
        elif attn_1_1_output in {"3"}:
            return attn_1_7_output == "</s>"

    num_attn_2_0_pattern = select(attn_1_7_outputs, attn_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_0_7_output):
        if position in {
            0,
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
            36,
            37,
            38,
            39,
            40,
            41,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return attn_0_7_output == ""
        elif position in {1, 2}:
            return attn_0_7_output == "<s>"
        elif position in {3, 5, 6}:
            return attn_0_7_output == "0"
        elif position in {4}:
            return attn_0_7_output == "</s>"
        elif position in {42, 35}:
            return attn_0_7_output == "<pad>"

    num_attn_2_1_pattern = select(attn_0_7_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_0_output, attn_1_6_output):
        if attn_1_0_output in {"</s>", "4", "0"}:
            return attn_1_6_output == ""
        elif attn_1_0_output in {"3", "1"}:
            return attn_1_6_output == "2"
        elif attn_1_0_output in {"2"}:
            return attn_1_6_output == "1"
        elif attn_1_0_output in {"<s>"}:
            return attn_1_6_output == "</s>"

    num_attn_2_2_pattern = select(attn_1_6_outputs, attn_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_2_output, attn_0_3_output):
        if mlp_0_2_output in {
            0,
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
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            40,
            41,
            42,
            43,
            44,
            45,
            47,
            48,
            49,
        }:
            return attn_0_3_output == ""
        elif mlp_0_2_output in {1, 31}:
            return attn_0_3_output == "0"
        elif mlp_0_2_output in {30}:
            return attn_0_3_output == "<pad>"
        elif mlp_0_2_output in {39}:
            return attn_0_3_output == "</s>"
        elif mlp_0_2_output in {46}:
            return attn_0_3_output == "<s>"

    num_attn_2_3_pattern = select(attn_0_3_outputs, mlp_0_2_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_5_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_0_output, attn_0_6_output):
        if attn_1_0_output in {"0"}:
            return attn_0_6_output == 6
        elif attn_1_0_output in {"1"}:
            return attn_0_6_output == 1
        elif attn_1_0_output in {"2"}:
            return attn_0_6_output == 7
        elif attn_1_0_output in {"3"}:
            return attn_0_6_output == 24
        elif attn_1_0_output in {"4"}:
            return attn_0_6_output == 28
        elif attn_1_0_output in {"</s>"}:
            return attn_0_6_output == 2
        elif attn_1_0_output in {"<s>"}:
            return attn_0_6_output == 39

    num_attn_2_4_pattern = select(attn_0_6_outputs, attn_1_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_1_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_0_output, num_mlp_1_1_output):
        if attn_1_0_output in {"0"}:
            return num_mlp_1_1_output == 9
        elif attn_1_0_output in {"1"}:
            return num_mlp_1_1_output == 2
        elif attn_1_0_output in {"2"}:
            return num_mlp_1_1_output == 44
        elif attn_1_0_output in {"3"}:
            return num_mlp_1_1_output == 14
        elif attn_1_0_output in {"4"}:
            return num_mlp_1_1_output == 19
        elif attn_1_0_output in {"</s>"}:
            return num_mlp_1_1_output == 16
        elif attn_1_0_output in {"<s>"}:
            return num_mlp_1_1_output == 29

    num_attn_2_5_pattern = select(
        num_mlp_1_1_outputs, attn_1_0_outputs, num_predicate_2_5
    )
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_4_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_0_output, attn_0_5_output):
        if attn_1_0_output in {"3", "0"}:
            return attn_0_5_output == "0"
        elif attn_1_0_output in {"4", "2", "1"}:
            return attn_0_5_output == ""
        elif attn_1_0_output in {"</s>"}:
            return attn_0_5_output == "</s>"
        elif attn_1_0_output in {"<s>"}:
            return attn_0_5_output == "1"

    num_attn_2_6_pattern = select(attn_0_5_outputs, attn_1_0_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_1_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_0_output, attn_0_4_output):
        if attn_1_0_output in {"0"}:
            return attn_0_4_output == "1"
        elif attn_1_0_output in {"3", "1"}:
            return attn_0_4_output == "</s>"
        elif attn_1_0_output in {"2"}:
            return attn_0_4_output == "0"
        elif attn_1_0_output in {"</s>", "<s>", "4"}:
            return attn_0_4_output == "2"

    num_attn_2_7_pattern = select(attn_0_4_outputs, attn_1_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_1_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_1_2_output, attn_2_0_output):
        key = (mlp_1_2_output, attn_2_0_output)
        return 17

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_1_2_outputs, attn_2_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(mlp_0_3_output, attn_1_1_output):
        key = (mlp_0_3_output, attn_1_1_output)
        return 23

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(mlp_0_3_outputs, attn_1_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(position, num_mlp_0_3_output):
        key = (position, num_mlp_0_3_output)
        return 2

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(positions, num_mlp_0_3_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_1_0_output, attn_2_2_output):
        key = (attn_1_0_output, attn_2_2_output)
        return 12

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_2_2_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_1_output):
        key = num_attn_1_1_output
        return 10

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_1_7_output):
        key = (num_attn_1_1_output, num_attn_1_7_output)
        return 2

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_3_output):
        key = num_attn_1_3_output
        return 2

    num_mlp_2_2_outputs = [num_mlp_2_2(k0) for k0 in num_attn_1_3_outputs]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_2_output, num_attn_1_0_output):
        key = (num_attn_1_2_output, num_attn_1_0_output)
        return 28

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_0_outputs)
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
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
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
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
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
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
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


print(run(["<s>", "1", "0", "0", "2", "1", "2", "</s>"]))
