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
        "output/length/rasp/mostfreq/trainlength40/s4/most_freq_weights.csv",
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
        if q_position in {0, 29}:
            return k_position == 4
        elif q_position in {1, 34}:
            return k_position == 1
        elif q_position in {2, 11, 12}:
            return k_position == 2
        elif q_position in {10, 3, 31}:
            return k_position == 3
        elif q_position in {8, 4, 39}:
            return k_position == 6
        elif q_position in {32, 5, 15}:
            return k_position == 11
        elif q_position in {25, 28, 6, 23}:
            return k_position == 16
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {9, 46}:
            return k_position == 8
        elif q_position in {18, 13, 38}:
            return k_position == 21
        elif q_position in {19, 14}:
            return k_position == 24
        elif q_position in {16}:
            return k_position == 23
        elif q_position in {17, 41}:
            return k_position == 14
        elif q_position in {20}:
            return k_position == 34
        elif q_position in {21, 22}:
            return k_position == 10
        elif q_position in {24}:
            return k_position == 12
        elif q_position in {26, 30}:
            return k_position == 20
        elif q_position in {27}:
            return k_position == 38
        elif q_position in {33}:
            return k_position == 31
        elif q_position in {40, 35}:
            return k_position == 18
        elif q_position in {36, 37}:
            return k_position == 19
        elif q_position in {42, 45}:
            return k_position == 48
        elif q_position in {43, 47}:
            return k_position == 35
        elif q_position in {44}:
            return k_position == 27
        elif q_position in {48}:
            return k_position == 44
        elif q_position in {49}:
            return k_position == 29

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 25, 3}:
            return token == "4"
        elif position in {1}:
            return token == "3"
        elif position in {
            2,
            4,
            5,
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
            return token == "2"
        elif position in {7}:
            return token == "5"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 24}:
            return token == ""

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 12, 13, 14, 17, 19, 23, 25, 26, 30}:
            return token == "4"
        elif position in {1, 7}:
            return token == "0"
        elif position in {2}:
            return token == "5"
        elif position in {3, 4, 5, 6, 9}:
            return token == "1"
        elif position in {
            8,
            10,
            11,
            15,
            16,
            18,
            20,
            21,
            22,
            24,
            27,
            28,
            29,
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
            return token == "2"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 30
        elif q_position in {48, 1}:
            return k_position == 32
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 18
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {5}:
            return k_position == 25
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {19, 7}:
            return k_position == 10
        elif q_position in {8, 10}:
            return k_position == 3
        elif q_position in {
            9,
            11,
            12,
            14,
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
            37,
            38,
        }:
            return k_position == 21
        elif q_position in {36, 13, 15, 16, 17, 18}:
            return k_position == 11
        elif q_position in {39}:
            return k_position == 27
        elif q_position in {40}:
            return k_position == 38
        elif q_position in {41}:
            return k_position == 36
        elif q_position in {42, 44}:
            return k_position == 20
        elif q_position in {43}:
            return k_position == 45
        elif q_position in {45}:
            return k_position == 8
        elif q_position in {46, 47}:
            return k_position == 28
        elif q_position in {49}:
            return k_position == 42

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 38}:
            return k_position == 2
        elif q_position in {1, 36}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {11, 3}:
            return k_position == 4
        elif q_position in {8, 33, 4}:
            return k_position == 5
        elif q_position in {10, 5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 22
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {9, 42}:
            return k_position == 6
        elif q_position in {12, 15}:
            return k_position == 33
        elif q_position in {13}:
            return k_position == 9
        elif q_position in {14}:
            return k_position == 36
        elif q_position in {16}:
            return k_position == 18
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18, 47}:
            return k_position == 27
        elif q_position in {25, 19, 20}:
            return k_position == 38
        elif q_position in {21}:
            return k_position == 11
        elif q_position in {22}:
            return k_position == 15
        elif q_position in {32, 23, 26, 27, 28, 30, 31}:
            return k_position == 0
        elif q_position in {24}:
            return k_position == 19
        elif q_position in {29}:
            return k_position == 23
        elif q_position in {34, 35, 37}:
            return k_position == 21
        elif q_position in {40, 39}:
            return k_position == 10
        elif q_position in {41}:
            return k_position == 47
        elif q_position in {43}:
            return k_position == 34
        elif q_position in {44}:
            return k_position == 12
        elif q_position in {45}:
            return k_position == 46
        elif q_position in {46}:
            return k_position == 31
        elif q_position in {48}:
            return k_position == 35
        elif q_position in {49}:
            return k_position == 32

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 8, 10, 11}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 39}:
            return k_position == 2
        elif q_position in {3, 4}:
            return k_position == 6
        elif q_position in {13, 21, 5}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {41, 7}:
            return k_position == 7
        elif q_position in {9, 43}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {17, 18, 15}:
            return k_position == 11
        elif q_position in {37, 16, 22, 26, 27, 31}:
            return k_position == 16
        elif q_position in {32, 33, 34, 35, 19, 20, 23, 24, 25, 29, 30}:
            return k_position == 22
        elif q_position in {28}:
            return k_position == 18
        elif q_position in {36}:
            return k_position == 34
        elif q_position in {45, 38}:
            return k_position == 15
        elif q_position in {40}:
            return k_position == 19
        elif q_position in {42}:
            return k_position == 31
        elif q_position in {44, 47}:
            return k_position == 45
        elif q_position in {46}:
            return k_position == 21
        elif q_position in {48}:
            return k_position == 48
        elif q_position in {49}:
            return k_position == 40

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, positions)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 3, 35, 11, 21}:
            return k_position == 2
        elif q_position in {1, 2, 36, 12, 14}:
            return k_position == 1
        elif q_position in {40, 4}:
            return k_position == 22
        elif q_position in {9, 5}:
            return k_position == 6
        elif q_position in {32, 25, 6}:
            return k_position == 0
        elif q_position in {29, 7}:
            return k_position == 5
        elif q_position in {8, 20}:
            return k_position == 4
        elif q_position in {10}:
            return k_position == 10
        elif q_position in {18, 13}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 26
        elif q_position in {16, 27}:
            return k_position == 16
        elif q_position in {17, 22}:
            return k_position == 9
        elif q_position in {19}:
            return k_position == 13
        elif q_position in {26, 23}:
            return k_position == 19
        elif q_position in {24}:
            return k_position == 14
        elif q_position in {28}:
            return k_position == 33
        elif q_position in {37, 30}:
            return k_position == 20
        elif q_position in {43, 31}:
            return k_position == 3
        elif q_position in {33}:
            return k_position == 34
        elif q_position in {34}:
            return k_position == 36
        elif q_position in {41, 42, 38}:
            return k_position == 35
        elif q_position in {45, 39}:
            return k_position == 21
        elif q_position in {44}:
            return k_position == 38
        elif q_position in {46}:
            return k_position == 23
        elif q_position in {47}:
            return k_position == 15
        elif q_position in {48}:
            return k_position == 31
        elif q_position in {49}:
            return k_position == 45

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 2}:
            return token == "1"
        elif position in {3, 4, 5, 6, 8}:
            return token == "5"
        elif position in {39, 7}:
            return token == "4"
        elif position in {
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
        }:
            return token == "2"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 1}:
            return token == "2"
        elif position in {2, 3, 4, 5, 6, 7}:
            return token == "<s>"
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
            24,
            25,
            26,
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
        }:
            return token == ""
        elif position in {27, 23}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 5, 6}:
            return token == "1"
        elif position in {1}:
            return token == "2"
        elif position in {2, 3, 4, 7}:
            return token == "<s>"
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
        }:
            return token == ""
        elif position in {26}:
            return token == "<pad>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {
            0,
            2,
            3,
            4,
            5,
            28,
            32,
            35,
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
        }:
            return token == ""
        elif position in {1}:
            return token == "1"
        elif position in {
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
            29,
            30,
            31,
            33,
            34,
            36,
            37,
            38,
        }:
            return token == "<s>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1, 2, 3, 4, 5, 6, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""
        elif position in {38, 7, 8, 11, 24, 29, 31}:
            return token == "4"
        elif position in {32, 33, 36, 9, 12, 13, 14, 15, 16, 17, 20, 21, 22, 28}:
            return token == "0"
        elif position in {10}:
            return token == "3"
        elif position in {35, 37, 18, 23, 30}:
            return token == "1"
        elif position in {27, 19}:
            return token == "5"
        elif position in {25, 26, 34, 39}:
            return token == "2"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 2, 3, 4, 5, 6, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""
        elif position in {1}:
            return token == "5"
        elif position in {8, 10, 11, 7}:
            return token == "0"
        elif position in {9}:
            return token == "1"
        elif position in {
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
        }:
            return token == "<s>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 1}:
            return token == "3"
        elif position in {
            2,
            3,
            4,
            5,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
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
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif position in {8, 6, 7}:
            return token == "<s>"
        elif position in {17, 18, 43, 44}:
            return token == "<pad>"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 1, 7}:
            return token == "4"
        elif position in {2, 3, 4}:
            return token == "<s>"
        elif position in {5, 6}:
            return token == "2"
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
        }:
            return token == ""
        elif position in {29}:
            return token == "<pad>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 1, 7, 8, 9, 10, 11}:
            return token == "0"
        elif position in {
            2,
            3,
            4,
            5,
            13,
            14,
            15,
            16,
            17,
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
        }:
            return token == ""
        elif position in {6}:
            return token == "<s>"
        elif position in {12, 20, 30}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_4_output, attn_0_1_output):
        key = (attn_0_4_output, attn_0_1_output)
        if key in {
            ("0", "3"),
            ("0", "5"),
            ("0", "<s>"),
            ("1", "3"),
            ("1", "5"),
            ("1", "<s>"),
            ("3", "3"),
            ("3", "5"),
            ("3", "<s>"),
            ("4", "3"),
            ("4", "5"),
            ("4", "<s>"),
            ("5", "3"),
            ("5", "5"),
            ("5", "<s>"),
            ("<s>", "3"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 31
        return 21

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_6_output, attn_0_0_output):
        key = (attn_0_6_output, attn_0_0_output)
        if key in {
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("1", "<s>"),
            ("2", "1"),
            ("4", "1"),
            ("5", "1"),
            ("<s>", "1"),
        }:
            return 4
        return 5

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_5_output, attn_0_7_output):
        key = (attn_0_5_output, attn_0_7_output)
        return 5

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_7_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_0_output, attn_0_6_output):
        key = (attn_0_0_output, attn_0_6_output)
        if key in {("0", "0"), ("2", "0"), ("4", "0")}:
            return 49
        return 15

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_6_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_7_output):
        key = (num_attn_0_0_output, num_attn_0_7_output)
        return 5

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_4_output):
        key = (num_attn_0_3_output, num_attn_0_4_output)
        return 8

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_7_output, num_attn_0_5_output):
        key = (num_attn_0_7_output, num_attn_0_5_output)
        return 0

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_4_output, num_attn_0_7_output):
        key = (num_attn_0_4_output, num_attn_0_7_output)
        return 1

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 33, 7, 14, 15, 18}:
            return token == "3"
        elif position in {1, 3, 8, 41, 10, 12, 17, 19, 28}:
            return token == ""
        elif position in {2, 45}:
            return token == "0"
        elif position in {32, 37, 4, 5}:
            return token == "<s>"
        elif position in {29, 6, 30}:
            return token == "1"
        elif position in {40, 9, 42, 43, 44, 13, 46, 47, 48, 20, 23, 24, 26}:
            return token == "2"
        elif position in {34, 35, 36, 39, 11, 16, 22, 25, 27, 31}:
            return token == "5"
        elif position in {49, 21, 38}:
            return token == "4"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {
            0,
            12,
            14,
            15,
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
            36,
            39,
        }:
            return token == "1"
        elif position in {1, 43, 46}:
            return token == "5"
        elif position in {2, 3, 35, 37, 40, 41, 47, 48, 17}:
            return token == "2"
        elif position in {4, 5, 38, 6, 10}:
            return token == "3"
        elif position in {7}:
            return token == "0"
        elif position in {8}:
            return token == "4"
        elif position in {9, 44, 45, 13, 49}:
            return token == ""
        elif position in {16, 11, 20}:
            return token == "<s>"
        elif position in {42}:
            return token == "<pad>"

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "<s>"
        elif q_token in {"<s>", "2"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "3"
        elif q_token in {"5"}:
            return k_token == "0"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 2, 3, 45}:
            return token == "3"
        elif position in {4, 5, 6}:
            return token == "4"
        elif position in {
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            16,
            18,
            20,
            21,
            22,
            23,
            25,
            26,
            27,
            28,
            29,
            31,
            33,
            34,
            35,
            36,
            38,
            39,
        }:
            return token == "2"
        elif position in {32, 37, 10, 15, 17, 19, 24, 30}:
            return token == "1"
        elif position in {40, 47}:
            return token == "5"
        elif position in {41, 42, 43, 44, 46, 48, 49}:
            return token == ""

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_6_output, token):
        if attn_0_6_output in {"0"}:
            return token == "2"
        elif attn_0_6_output in {"2", "1"}:
            return token == "4"
        elif attn_0_6_output in {"3"}:
            return token == "1"
        elif attn_0_6_output in {"4"}:
            return token == "5"
        elif attn_0_6_output in {"5"}:
            return token == ""
        elif attn_0_6_output in {"<s>"}:
            return token == "<s>"

    attn_1_4_pattern = select_closest(tokens, attn_0_6_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(token, position):
        if token in {"0"}:
            return position == 6
        elif token in {"3", "1"}:
            return position == 1
        elif token in {"2"}:
            return position == 30
        elif token in {"4"}:
            return position == 5
        elif token in {"5"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 10

    attn_1_5_pattern = select_closest(positions, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_5_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"4", "3", "1", "<s>"}:
            return k_token == ""
        elif q_token in {"2"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "0"

    attn_1_6_pattern = select_closest(tokens, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"2"}:
            return k_token == "<s>"
        elif q_token in {"3", "4"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_1_7_pattern = select_closest(tokens, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_1_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 1, 2, 3}:
            return token == "5"
        elif position in {4, 5, 6, 7, 9}:
            return token == "<s>"
        elif position in {
            8,
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
        }:
            return token == ""

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_4_output, attn_0_2_output):
        if attn_0_4_output in {"1", "5", "0", "<s>", "3", "4", "2"}:
            return attn_0_2_output == ""

    num_attn_1_1_pattern = select(attn_0_2_outputs, attn_0_4_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_7_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_6_output, attn_0_2_output):
        if attn_0_6_output in {"5", "1", "0", "3", "2"}:
            return attn_0_2_output == ""
        elif attn_0_6_output in {"<s>", "4"}:
            return attn_0_2_output == "4"

    num_attn_1_2_pattern = select(attn_0_2_outputs, attn_0_6_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_6_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1}:
            return token == "4"
        elif position in {
            2,
            3,
            4,
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
        }:
            return token == ""
        elif position in {35, 5}:
            return token == "<pad>"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_6_output, attn_0_7_output):
        if attn_0_6_output in {"<s>", "0"}:
            return attn_0_7_output == "0"
        elif attn_0_6_output in {"1", "5", "3", "4", "2"}:
            return attn_0_7_output == ""

    num_attn_1_4_pattern = select(attn_0_7_outputs, attn_0_6_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_7_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(token, position):
        if token in {"0"}:
            return position == 35
        elif token in {"1"}:
            return position == 33
        elif token in {"2"}:
            return position == 9
        elif token in {"3"}:
            return position == 7
        elif token in {"4"}:
            return position == 19
        elif token in {"5"}:
            return position == 49
        elif token in {"<s>"}:
            return position == 0

    num_attn_1_5_pattern = select(positions, tokens, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_5_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_1_output, attn_0_4_output):
        if attn_0_1_output in {"1", "5", "0", "3", "4"}:
            return attn_0_4_output == ""
        elif attn_0_1_output in {"2"}:
            return attn_0_4_output == "2"
        elif attn_0_1_output in {"<s>"}:
            return attn_0_4_output == "<pad>"

    num_attn_1_6_pattern = select(attn_0_4_outputs, attn_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_0_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, token):
        if position in {0, 1}:
            return token == "0"
        elif position in {32, 2, 23}:
            return token == "<pad>"
        elif position in {
            3,
            4,
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
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
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
        elif position in {41, 5, 6}:
            return token == "4"

    num_attn_1_7_pattern = select(tokens, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, ones)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(mlp_0_2_output):
        key = mlp_0_2_output
        return 45

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in mlp_0_2_outputs]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_4_output, attn_0_5_output):
        key = (attn_0_4_output, attn_0_5_output)
        return 18

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_5_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_0_7_output, num_mlp_0_0_output):
        key = (attn_0_7_output, num_mlp_0_0_output)
        return 28

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_0_7_outputs, num_mlp_0_0_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_2_output, attn_1_7_output):
        key = (attn_1_2_output, attn_1_7_output)
        return 31

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_7_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_4_output, num_attn_1_7_output):
        key = (num_attn_1_4_output, num_attn_1_7_output)
        return 37

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_1_0_output):
        key = (num_attn_1_2_output, num_attn_1_0_output)
        return 43

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_0_3_output, num_attn_1_5_output):
        key = (num_attn_0_3_output, num_attn_1_5_output)
        return 31

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 31

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, attn_1_4_output):
        if position in {0, 34, 3, 36, 40, 45, 30, 31}:
            return attn_1_4_output == "3"
        elif position in {1, 42, 43}:
            return attn_1_4_output == "5"
        elif position in {49, 2, 6}:
            return attn_1_4_output == "4"
        elif position in {
            4,
            5,
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
            29,
            35,
            37,
            47,
        }:
            return attn_1_4_output == "<s>"
        elif position in {32, 28, 38}:
            return attn_1_4_output == "2"
        elif position in {33, 39}:
            return attn_1_4_output == "1"
        elif position in {48, 41, 44, 46}:
            return attn_1_4_output == ""

    attn_2_0_pattern = select_closest(attn_1_4_outputs, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_4_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, attn_0_5_output):
        if position in {0, 8, 42, 49, 18}:
            return attn_0_5_output == 9
        elif position in {1}:
            return attn_0_5_output == 3
        elif position in {2, 44, 47, 17, 28}:
            return attn_0_5_output == 7
        elif position in {10, 3, 45}:
            return attn_0_5_output == 1
        elif position in {4, 5, 6, 7, 9, 12, 15, 25, 30, 31}:
            return attn_0_5_output == 6
        elif position in {16, 11, 46}:
            return attn_0_5_output == 2
        elif position in {32, 13}:
            return attn_0_5_output == 12
        elif position in {14}:
            return attn_0_5_output == 11
        elif position in {35, 26, 19, 22}:
            return attn_0_5_output == 13
        elif position in {43, 20}:
            return attn_0_5_output == 4
        elif position in {21}:
            return attn_0_5_output == 44
        elif position in {23}:
            return attn_0_5_output == 45
        elif position in {24, 34, 37, 39}:
            return attn_0_5_output == 5
        elif position in {27}:
            return attn_0_5_output == 21
        elif position in {29}:
            return attn_0_5_output == 30
        elif position in {33}:
            return attn_0_5_output == 18
        elif position in {41, 36}:
            return attn_0_5_output == 42
        elif position in {38}:
            return attn_0_5_output == 26
        elif position in {40}:
            return attn_0_5_output == 25
        elif position in {48}:
            return attn_0_5_output == 33

    attn_2_1_pattern = select_closest(attn_0_5_outputs, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_5_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"5", "1", "0", "<s>", "3"}:
            return k_token == "5"
        elif q_token in {"2", "4"}:
            return k_token == "0"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"5", "0"}:
            return k_token == "4"
        elif q_token in {"3", "1"}:
            return k_token == "0"
        elif q_token in {"<s>", "2"}:
            return k_token == ""
        elif q_token in {"4"}:
            return k_token == "<s>"

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_token, k_token):
        if q_token in {"4", "3", "1", "0"}:
            return k_token == "5"
        elif q_token in {"2", "5"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_4_pattern = select_closest(tokens, tokens, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_6_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_6_output, token):
        if attn_0_6_output in {"0"}:
            return token == "2"
        elif attn_0_6_output in {"5", "1", "<s>", "3", "2"}:
            return token == ""
        elif attn_0_6_output in {"4"}:
            return token == "3"

    attn_2_5_pattern = select_closest(tokens, attn_0_6_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_4_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_0_output, token):
        if attn_0_0_output in {"3", "5", "0"}:
            return token == "<s>"
        elif attn_0_0_output in {"<s>", "1"}:
            return token == ""
        elif attn_0_0_output in {"2"}:
            return token == "0"
        elif attn_0_0_output in {"4"}:
            return token == "5"

    attn_2_6_pattern = select_closest(tokens, attn_0_0_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_0_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "5"
        elif q_token in {"4", "3", "1"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"5"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    attn_2_7_pattern = select_closest(tokens, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, mlp_0_1_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, attn_0_6_output):
        if position in {
            0,
            3,
            4,
            5,
            6,
            33,
            34,
            35,
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
        }:
            return attn_0_6_output == ""
        elif position in {1, 36, 10, 11, 15, 16, 31}:
            return attn_0_6_output == "0"
        elif position in {17, 2, 14}:
            return attn_0_6_output == "4"
        elif position in {7, 9, 18, 20, 24, 27, 28}:
            return attn_0_6_output == "1"
        elif position in {32, 8, 13, 19, 21, 22, 25, 26, 29, 30}:
            return attn_0_6_output == "2"
        elif position in {12, 23}:
            return attn_0_6_output == "5"

    num_attn_2_0_pattern = select(attn_0_6_outputs, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_7_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, token):
        if position in {
            0,
            2,
            3,
            4,
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
        }:
            return token == ""
        elif position in {1}:
            return token == "0"
        elif position in {5, 6}:
            return token == "3"
        elif position in {17}:
            return token == "<pad>"

    num_attn_2_1_pattern = select(tokens, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, ones)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, attn_0_4_output):
        if position in {
            0,
            4,
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
        }:
            return attn_0_4_output == ""
        elif position in {1}:
            return attn_0_4_output == "0"
        elif position in {2}:
            return attn_0_4_output == "1"
        elif position in {26, 3, 37}:
            return attn_0_4_output == "<pad>"
        elif position in {5, 6}:
            return attn_0_4_output == "5"

    num_attn_2_2_pattern = select(attn_0_4_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_7_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_1_output, attn_0_4_output):
        if attn_1_1_output in {"5", "1", "0", "<s>", "4", "2"}:
            return attn_0_4_output == ""
        elif attn_1_1_output in {"3"}:
            return attn_0_4_output == "3"

    num_attn_2_3_pattern = select(attn_0_4_outputs, attn_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_5_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_7_output, attn_0_0_output):
        if attn_0_7_output in {"1", "0", "<s>", "3", "4", "2"}:
            return attn_0_0_output == ""
        elif attn_0_7_output in {"5"}:
            return attn_0_0_output == "5"

    num_attn_2_4_pattern = select(attn_0_0_outputs, attn_0_7_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_0_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_2_output, attn_0_4_output):
        if attn_0_2_output in {"5", "0", "<s>", "3", "4", "2"}:
            return attn_0_4_output == ""
        elif attn_0_2_output in {"1"}:
            return attn_0_4_output == "1"

    num_attn_2_5_pattern = select(attn_0_4_outputs, attn_0_2_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_1_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(position, attn_1_1_output):
        if position in {0, 1, 2, 3, 4, 5, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return attn_1_1_output == ""
        elif position in {36, 6, 39, 7, 10}:
            return attn_1_1_output == "5"
        elif position in {8, 13}:
            return attn_1_1_output == "3"
        elif position in {9, 29, 17}:
            return attn_1_1_output == "1"
        elif position in {35, 11, 14, 15, 20, 23, 25, 26, 27, 31}:
            return attn_1_1_output == "0"
        elif position in {32, 33, 34, 12, 16, 18, 22, 24, 28, 30}:
            return attn_1_1_output == "2"
        elif position in {37, 19, 21, 38}:
            return attn_1_1_output == "4"

    num_attn_2_6_pattern = select(attn_1_1_outputs, positions, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_0_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(position, attn_0_4_output):
        if position in {
            0,
            1,
            2,
            4,
            24,
            28,
            30,
            31,
            33,
            34,
            35,
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
        }:
            return attn_0_4_output == ""
        elif position in {3}:
            return attn_0_4_output == "<pad>"
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
            25,
            26,
            27,
            29,
            37,
            38,
        }:
            return attn_0_4_output == "0"
        elif position in {21}:
            return attn_0_4_output == "5"
        elif position in {32, 36}:
            return attn_0_4_output == "4"

    num_attn_2_7_pattern = select(attn_0_4_outputs, positions, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_0_output, position):
        key = (num_mlp_0_0_output, position)
        return 44

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, positions)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_7_output, mlp_1_0_output):
        key = (attn_2_7_output, mlp_1_0_output)
        return 5

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_7_outputs, mlp_1_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_7_output, num_mlp_1_3_output):
        key = (attn_2_7_output, num_mlp_1_3_output)
        return 27

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_7_outputs, num_mlp_1_3_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(mlp_1_1_output, attn_1_3_output):
        key = (mlp_1_1_output, attn_1_3_output)
        return 24

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(mlp_1_1_outputs, attn_1_3_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_7_output, num_attn_1_4_output):
        key = (num_attn_2_7_output, num_attn_1_4_output)
        return 4

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_5_output, num_attn_1_7_output):
        key = (num_attn_2_5_output, num_attn_1_7_output)
        return 0

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_5_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_1_output, num_attn_1_6_output):
        key = (num_attn_1_1_output, num_attn_1_6_output)
        return 3

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_6_output, num_attn_1_3_output):
        key = (num_attn_2_6_output, num_attn_1_3_output)
        return 2

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_6_outputs, num_attn_1_3_outputs)
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


print(run(["<s>", "1", "0", "0", "2", "1", "2"]))
