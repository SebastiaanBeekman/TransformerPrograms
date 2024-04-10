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
        "output/length/rasp/mostfreq/trainlength40/s3/most_freq_weights.csv",
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
    def predicate_0_0(position, token):
        if position in {0, 10, 12, 21}:
            return token == "3"
        elif position in {1, 33, 35, 5, 37, 39, 9, 25}:
            return token == "4"
        elif position in {2, 3}:
            return token == "2"
        elif position in {4, 13, 17, 18, 20, 29}:
            return token == "5"
        elif position in {34, 36, 6, 7, 11, 15, 23, 28}:
            return token == "1"
        elif position in {32, 38, 8, 22, 24, 26, 27, 31}:
            return token == "0"
        elif position in {40, 41, 42, 43, 44, 45, 14, 46, 16, 47, 48, 19, 49}:
            return token == ""
        elif position in {30}:
            return token == "<s>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 17, 10, 29}:
            return token == "3"
        elif position in {1, 33, 35, 38, 39, 12, 15, 19, 27}:
            return token == "4"
        elif position in {2, 3, 4, 5, 6, 37, 25, 26}:
            return token == "0"
        elif position in {32, 34, 7, 8, 9, 13, 18, 21, 22, 24, 28, 30, 31}:
            return token == "1"
        elif position in {16, 11, 20, 36}:
            return token == "5"
        elif position in {40, 41, 42, 43, 44, 45, 14, 46, 47, 48, 49}:
            return token == ""
        elif position in {23}:
            return token == "2"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 32, 6, 7, 12, 44, 21, 23, 25, 28}:
            return token == "5"
        elif position in {1, 34, 35, 36, 38, 22, 24, 27, 29, 30}:
            return token == "0"
        elif position in {16, 2}:
            return token == "2"
        elif position in {3, 4, 5, 37, 8, 13}:
            return token == "1"
        elif position in {9}:
            return token == "3"
        elif position in {33, 39, 10, 17, 18, 19, 20}:
            return token == "4"
        elif position in {40, 41, 42, 11, 43, 45, 14, 15, 46, 47, 48, 49, 26, 31}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 10}:
            return token == "3"
        elif position in {1, 2, 7, 8, 19, 24, 25, 26}:
            return token == "1"
        elif position in {3, 4, 5, 6, 37, 22, 29, 30}:
            return token == "0"
        elif position in {32, 35, 9, 15, 17, 20, 21, 31}:
            return token == "5"
        elif position in {33, 34, 36, 38, 39, 11, 16, 18, 23, 28}:
            return token == "4"
        elif position in {40, 41, 42, 43, 12, 13, 14, 44, 45, 46, 47, 48, 49, 27}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 49, 44}:
            return k_position == 8
        elif q_position in {1, 7}:
            return k_position == 4
        elif q_position in {2, 3, 4, 42, 46}:
            return k_position == 9
        elif q_position in {5}:
            return k_position == 20
        elif q_position in {6}:
            return k_position == 26
        elif q_position in {8, 9, 28}:
            return k_position == 2
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {32, 12, 13, 16, 17, 18, 22}:
            return k_position == 3
        elif q_position in {14}:
            return k_position == 14
        elif q_position in {
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            15,
            19,
            20,
            23,
            24,
            25,
            26,
            27,
            29,
            30,
        }:
            return k_position == 23
        elif q_position in {21}:
            return k_position == 19
        elif q_position in {31}:
            return k_position == 28
        elif q_position in {40}:
            return k_position == 44
        elif q_position in {41}:
            return k_position == 49
        elif q_position in {43}:
            return k_position == 27
        elif q_position in {45}:
            return k_position == 0
        elif q_position in {47}:
            return k_position == 39
        elif q_position in {48}:
            return k_position == 21

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, positions)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 32, 33, 37, 38, 14, 20, 21, 23, 25, 26, 31}:
            return token == "0"
        elif position in {1, 2, 7, 12, 15, 24, 30}:
            return token == "1"
        elif position in {3, 4, 5, 6}:
            return token == "2"
        elif position in {34, 35, 36, 39, 8, 22, 27, 28, 29}:
            return token == "4"
        elif position in {9}:
            return token == "3"
        elif position in {10, 11, 13, 16, 17, 18}:
            return token == "5"
        elif position in {19}:
            return token == "<pad>"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 1, 2, 38, 18, 28, 31}:
            return k_position == 1
        elif q_position in {34, 3, 8, 9, 24}:
            return k_position == 2
        elif q_position in {11, 4, 5}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {33, 36, 7, 12, 14, 20, 23}:
            return k_position == 5
        elif q_position in {10, 13, 46}:
            return k_position == 29
        elif q_position in {15}:
            return k_position == 8
        elif q_position in {16}:
            return k_position == 16
        elif q_position in {17}:
            return k_position == 24
        elif q_position in {19}:
            return k_position == 7
        elif q_position in {37, 21, 22, 25, 26}:
            return k_position == 6
        elif q_position in {32, 27, 30, 39}:
            return k_position == 0
        elif q_position in {48, 29}:
            return k_position == 39
        elif q_position in {35}:
            return k_position == 21
        elif q_position in {40}:
            return k_position == 18
        elif q_position in {41}:
            return k_position == 23
        elif q_position in {42, 45}:
            return k_position == 10
        elif q_position in {43}:
            return k_position == 37
        elif q_position in {44}:
            return k_position == 33
        elif q_position in {47}:
            return k_position == 13
        elif q_position in {49}:
            return k_position == 45

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 35}:
            return k_position == 2
        elif q_position in {8, 1, 2}:
            return k_position == 1
        elif q_position in {10, 3, 34}:
            return k_position == 4
        elif q_position in {24, 4}:
            return k_position == 6
        elif q_position in {11, 5, 14}:
            return k_position == 7
        elif q_position in {29, 6}:
            return k_position == 16
        elif q_position in {12, 39, 7}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {16, 13}:
            return k_position == 9
        elif q_position in {21, 15}:
            return k_position == 11
        elif q_position in {17, 33}:
            return k_position == 26
        elif q_position in {18, 30}:
            return k_position == 18
        elif q_position in {19}:
            return k_position == 22
        elif q_position in {20}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 14
        elif q_position in {23}:
            return k_position == 19
        elif q_position in {25}:
            return k_position == 21
        elif q_position in {26}:
            return k_position == 35
        elif q_position in {27}:
            return k_position == 13
        elif q_position in {28, 46}:
            return k_position == 28
        elif q_position in {31}:
            return k_position == 8
        elif q_position in {32}:
            return k_position == 30
        elif q_position in {36}:
            return k_position == 29
        elif q_position in {37}:
            return k_position == 31
        elif q_position in {38}:
            return k_position == 34
        elif q_position in {40}:
            return k_position == 39
        elif q_position in {41}:
            return k_position == 43
        elif q_position in {42}:
            return k_position == 38
        elif q_position in {43}:
            return k_position == 41
        elif q_position in {44}:
            return k_position == 27
        elif q_position in {45}:
            return k_position == 44
        elif q_position in {47}:
            return k_position == 15
        elif q_position in {48}:
            return k_position == 42
        elif q_position in {49}:
            return k_position == 48

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 7, 8, 9, 10, 11}:
            return token == "<s>"
        elif position in {1}:
            return token == "4"
        elif position in {
            2,
            3,
            4,
            12,
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
        elif position in {5, 6}:
            return token == "2"
        elif position in {20, 28}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1}:
            return token == "4"
        elif position in {
            2,
            3,
            9,
            10,
            11,
            12,
            13,
            15,
            16,
            17,
            18,
            19,
            20,
            23,
            25,
            26,
            27,
            29,
            30,
            31,
            34,
            35,
            37,
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
        elif position in {8, 4, 21, 7}:
            return token == "<s>"
        elif position in {32, 33, 5, 6}:
            return token == "0"
        elif position in {38, 14}:
            return token == "2"
        elif position in {24, 22}:
            return token == "5"
        elif position in {28, 36}:
            return token == "<pad>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 1, 2, 3, 4, 5, 6, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""
        elif position in {
            33,
            34,
            35,
            37,
            7,
            8,
            9,
            39,
            12,
            14,
            15,
            20,
            21,
            22,
            24,
            26,
            27,
            29,
        }:
            return token == "1"
        elif position in {32, 10, 11, 13, 16, 17, 18, 19, 30}:
            return token == "3"
        elif position in {23}:
            return token == "4"
        elif position in {25, 36, 31}:
            return token == "5"
        elif position in {28, 38}:
            return token == "0"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {
            0,
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
            20,
            21,
            24,
            25,
            26,
            27,
            29,
            30,
            31,
            33,
            35,
            36,
            38,
        }:
            return token == "<s>"
        elif position in {1}:
            return token == "5"
        elif position in {
            2,
            3,
            4,
            5,
            16,
            17,
            18,
            19,
            23,
            28,
            32,
            34,
            37,
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
        elif position in {22}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {
            0,
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
            19,
            20,
            22,
            24,
            26,
            27,
            29,
            44,
        }:
            return token == "<s>"
        elif position in {1}:
            return token == "0"
        elif position in {
            2,
            3,
            4,
            5,
            18,
            21,
            23,
            25,
            28,
            30,
            31,
            33,
            34,
            35,
            36,
            37,
            39,
            40,
            41,
            43,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif position in {32, 42}:
            return token == "<pad>"
        elif position in {38}:
            return token == "2"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {
            0,
            1,
            2,
            3,
            4,
            5,
            18,
            27,
            28,
            29,
            31,
            32,
            33,
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
        elif position in {
            6,
            7,
            8,
            9,
            10,
            12,
            13,
            14,
            15,
            16,
            17,
            20,
            21,
            22,
            23,
            24,
            26,
            30,
            34,
        }:
            return token == "<s>"
        elif position in {19, 25, 11}:
            return token == "3"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {
            0,
            1,
            2,
            3,
            4,
            11,
            15,
            17,
            18,
            19,
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
            32,
            33,
            34,
            35,
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
        elif position in {37, 36, 5}:
            return token == "<pad>"
        elif position in {6}:
            return token == "0"
        elif position in {7, 8, 13, 14, 16, 24, 30}:
            return token == "<s>"
        elif position in {9, 10, 12}:
            return token == "3"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 1}:
            return token == "2"
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
        elif position in {8, 6, 7}:
            return token == "<s>"
        elif position in {26, 14}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_6_output, attn_0_5_output):
        key = (attn_0_6_output, attn_0_5_output)
        return 16

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_5_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, attn_0_6_output):
        key = (attn_0_1_output, attn_0_6_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "5"),
            ("0", "<s>"),
            ("1", "0"),
            ("2", "0"),
            ("3", "0"),
            ("4", "0"),
            ("5", "0"),
            ("<s>", "0"),
        }:
            return 3
        return 2

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_6_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(token, attn_0_5_output):
        key = (token, attn_0_5_output)
        return 47

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(tokens, attn_0_5_outputs)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_5_output, attn_0_2_output):
        key = (attn_0_5_output, attn_0_2_output)
        if key in {
            ("0", "5"),
            ("1", "5"),
            ("2", "5"),
            ("3", "5"),
            ("4", "5"),
            ("5", "0"),
            ("5", "1"),
            ("5", "2"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("5", "<s>"),
            ("<s>", "5"),
        }:
            return 18
        return 49

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_2_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output, num_attn_0_6_output):
        key = (num_attn_0_5_output, num_attn_0_6_output)
        return 13

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_7_output):
        key = (num_attn_0_0_output, num_attn_0_7_output)
        return 7

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 36

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_3_output, num_attn_0_7_output):
        key = (num_attn_0_3_output, num_attn_0_7_output)
        return 20

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_token, k_token):
        if q_token in {"<s>", "3", "0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "<s>"
        elif q_token in {"4", "2", "5"}:
            return k_token == ""

    attn_1_0_pattern = select_closest(tokens, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"3", "1", "2", "0"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "2"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"1", "4", "2", "5", "0"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == "0"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"0"}:
            return position == 2
        elif token in {"1", "2"}:
            return position == 7
        elif token in {"3"}:
            return position == 10
        elif token in {"4"}:
            return position == 1
        elif token in {"5"}:
            return position == 20
        elif token in {"<s>"}:
            return position == 15

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_7_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(token, position):
        if token in {"0"}:
            return position == 2
        elif token in {"1", "5"}:
            return position == 5
        elif token in {"2"}:
            return position == 4
        elif token in {"3"}:
            return position == 3
        elif token in {"4"}:
            return position == 40
        elif token in {"<s>"}:
            return position == 30

    attn_1_4_pattern = select_closest(positions, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_3_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, mlp_0_3_output):
        if position in {0, 36, 8, 26, 31}:
            return mlp_0_3_output == 18
        elif position in {1, 5, 6, 10, 14, 49}:
            return mlp_0_3_output == 6
        elif position in {40, 33, 2, 45}:
            return mlp_0_3_output == 15
        elif position in {48, 3, 4}:
            return mlp_0_3_output == 7
        elif position in {22, 7}:
            return mlp_0_3_output == 25
        elif position in {9}:
            return mlp_0_3_output == 12
        elif position in {11}:
            return mlp_0_3_output == 34
        elif position in {16, 32, 12, 15}:
            return mlp_0_3_output == 10
        elif position in {43, 13, 27, 28, 29}:
            return mlp_0_3_output == 21
        elif position in {17, 19, 21}:
            return mlp_0_3_output == 11
        elif position in {18}:
            return mlp_0_3_output == 24
        elif position in {20}:
            return mlp_0_3_output == 9
        elif position in {23}:
            return mlp_0_3_output == 37
        elif position in {24}:
            return mlp_0_3_output == 20
        elif position in {25}:
            return mlp_0_3_output == 28
        elif position in {30}:
            return mlp_0_3_output == 22
        elif position in {34, 39}:
            return mlp_0_3_output == 5
        elif position in {35}:
            return mlp_0_3_output == 3
        elif position in {37}:
            return mlp_0_3_output == 17
        elif position in {44, 38}:
            return mlp_0_3_output == 19
        elif position in {41, 46}:
            return mlp_0_3_output == 33
        elif position in {42}:
            return mlp_0_3_output == 48
        elif position in {47}:
            return mlp_0_3_output == 42

    attn_1_5_pattern = select_closest(mlp_0_3_outputs, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, mlp_0_3_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, attn_0_6_output):
        if position in {0, 32, 19, 22, 26, 28, 29, 30}:
            return attn_0_6_output == "<s>"
        elif position in {1, 20}:
            return attn_0_6_output == "3"
        elif position in {2, 10, 36, 14}:
            return attn_0_6_output == "2"
        elif position in {34, 3, 15, 17, 24}:
            return attn_0_6_output == "0"
        elif position in {35, 4, 8, 18, 23, 27, 31}:
            return attn_0_6_output == "1"
        elif position in {37, 38, 5, 6, 39, 40, 41, 43, 45, 46, 47, 48}:
            return attn_0_6_output == ""
        elif position in {33, 7, 42, 11, 13}:
            return attn_0_6_output == "5"
        elif position in {9, 12, 44, 16, 49, 21, 25}:
            return attn_0_6_output == "4"

    attn_1_6_pattern = select_closest(attn_0_6_outputs, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_7_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, token):
        if position in {0, 31}:
            return token == "5"
        elif position in {1, 34, 11, 22}:
            return token == "2"
        elif position in {33, 2, 36, 46, 49, 19}:
            return token == "<s>"
        elif position in {3, 7, 9, 43, 13, 45, 48, 18}:
            return token == ""
        elif position in {4, 37, 6, 39, 17, 21, 27, 29}:
            return token == "1"
        elif position in {5, 8, 40, 12, 16, 20, 25}:
            return token == "4"
        elif position in {41, 10, 42, 44, 14, 47, 26}:
            return token == "0"
        elif position in {32, 35, 38, 15, 23, 24, 28, 30}:
            return token == "3"

    attn_1_7_pattern = select_closest(tokens, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_6_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_1_output, attn_0_6_output):
        if num_mlp_0_1_output in {
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
            31,
            32,
            33,
            35,
            37,
            38,
            39,
            42,
            44,
            45,
            46,
            47,
            49,
        }:
            return attn_0_6_output == ""
        elif num_mlp_0_1_output in {34, 36, 40, 43, 48, 17, 18, 21, 30}:
            return attn_0_6_output == "<pad>"
        elif num_mlp_0_1_output in {41}:
            return attn_0_6_output == "<s>"

    num_attn_1_0_pattern = select(
        attn_0_6_outputs, num_mlp_0_1_outputs, num_predicate_1_0
    )
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_7_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
        if position in {0, 3, 4, 7, 8}:
            return token == "<s>"
        elif position in {1}:
            return token == "4"
        elif position in {
            2,
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
        }:
            return token == ""
        elif position in {5, 6}:
            return token == "1"

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0, 1}:
            return token == "3"
        elif position in {
            2,
            3,
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
        elif position in {4, 5, 7}:
            return token == "<s>"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0, 8, 6}:
            return token == "<s>"
        elif position in {1, 9, 7}:
            return token == "3"
        elif position in {
            2,
            3,
            4,
            5,
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
        elif position in {20}:
            return token == "<pad>"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(token, position):
        if token in {"4", "0"}:
            return position == 7
        elif token in {"1"}:
            return position == 41
        elif token in {"2"}:
            return position == 43
        elif token in {"3", "5"}:
            return position == 42
        elif token in {"<s>"}:
            return position == 8

    num_attn_1_4_pattern = select(positions, tokens, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_7_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, token):
        if position in {
            0,
            3,
            4,
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
        elif position in {2}:
            return token == "<pad>"
        elif position in {5, 6}:
            return token == "4"
        elif position in {8, 23, 7}:
            return token == "<s>"

    num_attn_1_5_pattern = select(tokens, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, ones)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(token, position):
        if token in {"1", "5", "0"}:
            return position == 7
        elif token in {"2"}:
            return position == 44
        elif token in {"3"}:
            return position == 47
        elif token in {"4"}:
            return position == 42
        elif token in {"<s>"}:
            return position == 10

    num_attn_1_6_pattern = select(positions, tokens, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_7_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(token, position):
        if token in {"0"}:
            return position == 9
        elif token in {"1", "3"}:
            return position == 7
        elif token in {"2"}:
            return position == 46
        elif token in {"4"}:
            return position == 41
        elif token in {"5"}:
            return position == 44
        elif token in {"<s>"}:
            return position == 35

    num_attn_1_7_pattern = select(positions, tokens, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_7_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_1_output, num_mlp_0_3_output):
        key = (attn_0_1_output, num_mlp_0_3_output)
        return 35

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, num_mlp_0_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position):
        key = position
        if key in {0, 3, 4, 5, 6, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return 27
        elif key in {1, 2}:
            return 2
        return 38

    mlp_1_1_outputs = [mlp_1_1(k0) for k0 in positions]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_3_output, attn_1_7_output):
        key = (attn_1_3_output, attn_1_7_output)
        if key in {
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "<s>"),
            ("<s>", "4"),
        }:
            return 2
        elif key in {
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("1", "<s>"),
        }:
            return 4
        return 22

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_7_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(mlp_0_2_output, attn_1_1_output):
        key = (mlp_0_2_output, attn_1_1_output)
        if key in {(25, "2"), (25, "3"), (25, "5")}:
            return 8
        return 13

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, attn_1_1_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_4_output, num_attn_1_0_output):
        key = (num_attn_1_4_output, num_attn_1_0_output)
        return 4

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_1_3_output):
        key = (num_attn_1_2_output, num_attn_1_3_output)
        return 36

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_0_output):
        key = num_attn_1_0_output
        return 15

    num_mlp_1_2_outputs = [num_mlp_1_2(k0) for k0 in num_attn_1_0_outputs]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_4_output, num_attn_1_3_output):
        key = (num_attn_1_4_output, num_attn_1_3_output)
        return 28

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "4"
        elif attn_0_3_output in {"1", "4", "3", "5"}:
            return token == "0"
        elif attn_0_3_output in {"2"}:
            return token == ""
        elif attn_0_3_output in {"<s>"}:
            return token == "3"

    attn_2_0_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"1", "2", "5"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_1_pattern = select_closest(tokens, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_4_output, token):
        if attn_0_4_output in {0, 4, 9, 11, 20, 26}:
            return token == "5"
        elif attn_0_4_output in {
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            10,
            12,
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
            25,
            28,
            29,
            30,
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
        elif attn_0_4_output in {17}:
            return token == "1"
        elif attn_0_4_output in {27}:
            return token == "3"
        elif attn_0_4_output in {32, 31}:
            return token == "2"

    attn_2_2_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"5", "2", "0"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"4", "3"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(mlp_1_1_output, mlp_1_2_output):
        if mlp_1_1_output in {0, 32, 36, 38, 39, 9, 11, 46, 15, 17, 18, 28}:
            return mlp_1_2_output == 44
        elif mlp_1_1_output in {1, 34, 6, 7, 10, 49, 21, 22, 30}:
            return mlp_1_2_output == 9
        elif mlp_1_1_output in {2, 47}:
            return mlp_1_2_output == 22
        elif mlp_1_1_output in {3, 12}:
            return mlp_1_2_output == 37
        elif mlp_1_1_output in {4}:
            return mlp_1_2_output == 11
        elif mlp_1_1_output in {33, 5}:
            return mlp_1_2_output == 17
        elif mlp_1_1_output in {
            8,
            13,
            14,
            16,
            19,
            20,
            23,
            24,
            25,
            26,
            27,
            29,
            35,
            37,
            40,
            41,
            42,
            43,
            44,
            45,
            48,
        }:
            return mlp_1_2_output == 5
        elif mlp_1_1_output in {31}:
            return mlp_1_2_output == 27

    attn_2_4_pattern = select_closest(mlp_1_2_outputs, mlp_1_1_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, mlp_1_2_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(mlp_1_1_output, mlp_0_1_output):
        if mlp_1_1_output in {0, 32, 33, 4, 6, 39, 10, 11, 12, 45, 30}:
            return mlp_0_1_output == 8
        elif mlp_1_1_output in {1, 5}:
            return mlp_0_1_output == 46
        elif mlp_1_1_output in {
            2,
            7,
            9,
            13,
            14,
            15,
            16,
            18,
            19,
            22,
            24,
            26,
            29,
            34,
            35,
            37,
            40,
            41,
            43,
            44,
            46,
            49,
        }:
            return mlp_0_1_output == 21
        elif mlp_1_1_output in {3}:
            return mlp_0_1_output == 38
        elif mlp_1_1_output in {36, 8, 42, 47, 17, 25}:
            return mlp_0_1_output == 37
        elif mlp_1_1_output in {48, 20, 21}:
            return mlp_0_1_output == 42
        elif mlp_1_1_output in {23}:
            return mlp_0_1_output == 24
        elif mlp_1_1_output in {27, 28, 38, 31}:
            return mlp_0_1_output == 0

    attn_2_5_pattern = select_closest(mlp_0_1_outputs, mlp_1_1_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, mlp_0_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "<s>"
        elif q_token in {"3", "2", "5"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_0_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(token, mlp_1_1_output):
        if token in {"0"}:
            return mlp_1_1_output == 16
        elif token in {"1"}:
            return mlp_1_1_output == 33
        elif token in {"2"}:
            return mlp_1_1_output == 13
        elif token in {"3"}:
            return mlp_1_1_output == 46
        elif token in {"<s>", "4", "5"}:
            return mlp_1_1_output == 7

    attn_2_7_pattern = select_closest(mlp_1_1_outputs, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_5_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_2_output, attn_1_6_output):
        if attn_0_2_output in {"1", "4", "2", "<s>", "5", "0"}:
            return attn_1_6_output == ""
        elif attn_0_2_output in {"3"}:
            return attn_1_6_output == "3"

    num_attn_2_0_pattern = select(attn_1_6_outputs, attn_0_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_5_output, attn_0_0_output):
        if attn_0_5_output in {"4", "3", "2", "<s>", "5", "0"}:
            return attn_0_0_output == ""
        elif attn_0_5_output in {"1"}:
            return attn_0_0_output == "3"

    num_attn_2_1_pattern = select(attn_0_0_outputs, attn_0_5_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_6_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, token):
        if position in {
            0,
            2,
            3,
            6,
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
            return token == "1"
        elif position in {4}:
            return token == "<pad>"
        elif position in {5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19}:
            return token == "<s>"

    num_attn_2_2_pattern = select(tokens, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, attn_1_6_output):
        if position in {
            0,
            2,
            3,
            4,
            7,
            8,
            9,
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
            return attn_1_6_output == ""
        elif position in {1}:
            return attn_1_6_output == "4"
        elif position in {5, 6}:
            return attn_1_6_output == "5"
        elif position in {10, 11}:
            return attn_1_6_output == "3"
        elif position in {36}:
            return attn_1_6_output == "<pad>"

    num_attn_2_3_pattern = select(attn_1_6_outputs, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(position, attn_0_2_output):
        if position in {
            0,
            1,
            2,
            3,
            4,
            5,
            8,
            11,
            12,
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
            42,
            44,
            45,
            46,
            47,
        }:
            return attn_0_2_output == ""
        elif position in {6, 7, 9, 10, 41, 43, 13, 17, 18, 49}:
            return attn_0_2_output == "3"
        elif position in {48}:
            return attn_0_2_output == "5"

    num_attn_2_4_pattern = select(attn_0_2_outputs, positions, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_3_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(position, attn_1_7_output):
        if position in {
            0,
            1,
            2,
            3,
            8,
            10,
            12,
            13,
            14,
            15,
            16,
            17,
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
            return attn_1_7_output == ""
        elif position in {4, 5, 6, 11, 18}:
            return attn_1_7_output == "3"
        elif position in {9, 7}:
            return attn_1_7_output == "4"

    num_attn_2_5_pattern = select(attn_1_7_outputs, positions, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_2_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_5_output, attn_1_7_output):
        if attn_0_5_output in {"1", "4", "3", "<s>", "5", "0"}:
            return attn_1_7_output == ""
        elif attn_0_5_output in {"2"}:
            return attn_1_7_output == "2"

    num_attn_2_6_pattern = select(attn_1_7_outputs, attn_0_5_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_7_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(token, position):
        if token in {"0"}:
            return position == 37
        elif token in {"1", "<s>"}:
            return position == 41
        elif token in {"3", "2"}:
            return position == 7
        elif token in {"4"}:
            return position == 35
        elif token in {"5"}:
            return position == 6

    num_attn_2_7_pattern = select(positions, tokens, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_0_1_output, attn_0_1_output):
        key = (mlp_0_1_output, attn_0_1_output)
        return 28

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, attn_0_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(mlp_1_3_output, attn_1_3_output):
        key = (mlp_1_3_output, attn_1_3_output)
        if key in {(36, "<s>"), (40, "<s>")}:
            return 13
        return 0

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(mlp_1_3_outputs, attn_1_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(mlp_1_3_output, mlp_1_0_output):
        key = (mlp_1_3_output, mlp_1_0_output)
        if key in {(0, 8), (0, 46), (23, 8), (25, 8), (36, 8), (46, 8)}:
            return 17
        elif key in {(41, 24)}:
            return 19
        return 12

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(mlp_1_3_outputs, mlp_1_0_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_1_1_output, mlp_0_0_output):
        key = (num_mlp_1_1_output, mlp_0_0_output)
        return 46

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_1_1_outputs, mlp_0_0_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_3_output, num_attn_1_0_output):
        key = (num_attn_2_3_output, num_attn_1_0_output)
        return 8

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_7_output, num_attn_1_6_output):
        key = (num_attn_1_7_output, num_attn_1_6_output)
        return 7

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_6_output, num_attn_1_2_output):
        key = (num_attn_2_6_output, num_attn_1_2_output)
        return 34

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_6_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_1_output, num_attn_1_0_output):
        key = (num_attn_1_1_output, num_attn_1_0_output)
        return 16

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_0_outputs)
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


print(
    run(
        [
            "<s>",
            "1",
            "3",
            "0",
            "0",
            "0",
            "5",
            "5",
            "3",
            "2",
            "3",
            "1",
            "1",
            "2",
            "5",
            "0",
            "4",
            "4",
            "5",
            "0",
            "2",
            "1",
            "2",
            "2",
            "2",
            "4",
        ]
    )
)
