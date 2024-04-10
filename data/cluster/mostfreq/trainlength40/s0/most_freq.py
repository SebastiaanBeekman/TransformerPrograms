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
        "output/length/rasp/mostfreq/trainlength40/s0/most_freq_weights.csv",
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
        if q_position in {0, 3, 11, 14, 16, 17, 25}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 30
        elif q_position in {40, 26, 4, 38}:
            return k_position == 7
        elif q_position in {49, 45, 20, 5}:
            return k_position == 22
        elif q_position in {6}:
            return k_position == 28
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {33, 34, 35, 36, 8}:
            return k_position == 25
        elif q_position in {9, 30, 15}:
            return k_position == 9
        elif q_position in {10}:
            return k_position == 21
        elif q_position in {12, 31}:
            return k_position == 16
        elif q_position in {18, 21, 13}:
            return k_position == 8
        elif q_position in {24, 19, 22}:
            return k_position == 23
        elif q_position in {28, 23}:
            return k_position == 26
        elif q_position in {27, 39}:
            return k_position == 10
        elif q_position in {29}:
            return k_position == 6
        elif q_position in {32}:
            return k_position == 39
        elif q_position in {37}:
            return k_position == 18
        elif q_position in {41, 43}:
            return k_position == 46
        elif q_position in {42, 46}:
            return k_position == 17
        elif q_position in {44}:
            return k_position == 48
        elif q_position in {47}:
            return k_position == 45
        elif q_position in {48}:
            return k_position == 38

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 6, 7, 8, 10, 11, 13, 15}:
            return token == "0"
        elif position in {1, 12, 16, 22, 26, 29}:
            return token == "3"
        elif position in {33, 2, 36, 38, 39, 24, 30, 31}:
            return token == "4"
        elif position in {34, 3, 37, 23, 28}:
            return token == "5"
        elif position in {32, 35, 4, 5, 18, 20, 21, 27}:
            return token == "2"
        elif position in {9, 47}:
            return token == "1"
        elif position in {40, 41, 42, 43, 44, 45, 14, 46, 48, 17, 49, 19, 25}:
            return token == ""

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 10, 14, 15, 16, 19, 21, 22, 27}:
            return token == "3"
        elif position in {1, 11}:
            return token == "0"
        elif position in {33, 2, 3, 38, 12, 17, 28, 31}:
            return token == "4"
        elif position in {
            32,
            34,
            35,
            4,
            5,
            6,
            36,
            37,
            9,
            18,
            20,
            23,
            24,
            25,
            26,
            29,
            30,
        }:
            return token == "5"
        elif position in {39, 7}:
            return token == "2"
        elif position in {8, 47}:
            return token == "1"
        elif position in {40, 41, 42, 43, 44, 13, 45, 46, 48, 49}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"0", "5", "2"}:
            return k_token == "4"
        elif q_token in {"1", "4", "3"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == "1"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 2, 11, 14}:
            return k_position == 1
        elif q_position in {3, 7}:
            return k_position == 4
        elif q_position in {8, 4}:
            return k_position == 5
        elif q_position in {10, 36, 5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 14
        elif q_position in {9, 41}:
            return k_position == 21
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {35, 13}:
            return k_position == 8
        elif q_position in {15}:
            return k_position == 6
        elif q_position in {16}:
            return k_position == 12
        elif q_position in {17, 27}:
            return k_position == 35
        elif q_position in {18}:
            return k_position == 32
        elif q_position in {19}:
            return k_position == 17
        elif q_position in {42, 20}:
            return k_position == 11
        elif q_position in {21}:
            return k_position == 3
        elif q_position in {38, 40, 22}:
            return k_position == 9
        elif q_position in {23}:
            return k_position == 24
        elif q_position in {24, 28}:
            return k_position == 20
        elif q_position in {25}:
            return k_position == 13
        elif q_position in {26}:
            return k_position == 23
        elif q_position in {29}:
            return k_position == 36
        elif q_position in {30}:
            return k_position == 10
        elif q_position in {32, 31}:
            return k_position == 26
        elif q_position in {33, 46}:
            return k_position == 34
        elif q_position in {34}:
            return k_position == 39
        elif q_position in {37}:
            return k_position == 25
        elif q_position in {39}:
            return k_position == 22
        elif q_position in {43}:
            return k_position == 29
        elif q_position in {44}:
            return k_position == 38
        elif q_position in {45}:
            return k_position == 47
        elif q_position in {47}:
            return k_position == 42
        elif q_position in {48}:
            return k_position == 40
        elif q_position in {49}:
            return k_position == 18

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {
            0,
            1,
            2,
            3,
            8,
            9,
            10,
            11,
            15,
            16,
            18,
            23,
            24,
            28,
            29,
            30,
            31,
            35,
            36,
            39,
        }:
            return token == "3"
        elif position in {32, 4, 5, 6, 38, 41, 21, 27}:
            return token == "2"
        elif position in {17, 12, 7}:
            return token == "4"
        elif position in {40, 42, 43, 44, 13, 46, 47, 48, 49, 19}:
            return token == ""
        elif position in {33, 34, 37, 14, 20, 22, 25, 26}:
            return token == "5"
        elif position in {45}:
            return token == "0"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {
            0,
            7,
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
            24,
            30,
            31,
            37,
        }:
            return token == "3"
        elif position in {1, 39}:
            return token == "5"
        elif position in {32, 33, 2, 3, 4, 5, 6, 34, 35, 36, 38, 22, 26, 27, 28, 29}:
            return token == "4"
        elif position in {8, 41, 40, 43, 42, 44, 45, 46, 47, 48, 49}:
            return token == ""
        elif position in {25, 23}:
            return token == "2"

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0, 48}:
            return token == "<s>"
        elif position in {1, 39}:
            return token == "5"
        elif position in {2, 34, 9, 14, 15, 16, 17, 18, 20, 21, 25, 27, 28, 30, 31}:
            return token == "0"
        elif position in {3, 4, 5, 6}:
            return token == "1"
        elif position in {32, 33, 37, 7, 10, 13, 24}:
            return token == "2"
        elif position in {36, 38, 8, 12, 22, 23, 29}:
            return token == "4"
        elif position in {19, 35, 11}:
            return token == "3"
        elif position in {40, 41, 42, 43, 45, 46, 47, 49, 26}:
            return token == ""
        elif position in {44}:
            return token == "<pad>"

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, positions)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 1, 3, 4, 6, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""
        elif position in {2}:
            return token == "4"
        elif position in {5}:
            return token == "<pad>"
        elif position in {32, 33, 34, 35, 7, 17, 18, 21, 22, 29, 30, 31}:
            return token == "2"
        elif position in {8, 12, 37, 39}:
            return token == "0"
        elif position in {36, 9, 13, 14, 15, 19, 20, 24, 26, 27}:
            return token == "3"
        elif position in {38, 10, 23, 25, 28}:
            return token == "5"
        elif position in {16, 11}:
            return token == "1"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 1}:
            return token == "5"
        elif position in {
            2,
            3,
            4,
            5,
            9,
            10,
            11,
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
        elif position in {8, 6, 7}:
            return token == "<s>"
        elif position in {12, 14}:
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
            21,
            23,
            24,
            25,
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
        elif position in {1}:
            return token == "1"
        elif position in {6}:
            return token == "5"
        elif position in {
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
            26,
            30,
        }:
            return token == "<s>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1}:
            return token == "3"
        elif position in {2, 3, 4, 5, 9}:
            return token == "<s>"
        elif position in {6}:
            return token == "2"
        elif position in {
            7,
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
        elif position in {31}:
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
            1,
            2,
            3,
            4,
            23,
            28,
            29,
            31,
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
        }:
            return token == ""
        elif position in {32, 5, 7, 8, 11, 14, 15, 17, 19, 20, 22, 24, 25, 26, 27, 30}:
            return token == "<s>"
        elif position in {16, 10, 18, 6}:
            return token == "1"
        elif position in {9, 21, 12, 13}:
            return token == "0"
        elif position in {39}:
            return token == "2"
        elif position in {49}:
            return token == "<pad>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 1}:
            return token == "0"
        elif position in {2, 3, 4, 5}:
            return token == "<s>"
        elif position in {6}:
            return token == "5"
        elif position in {
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
        }:
            return token == ""

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 1, 2}:
            return token == "4"
        elif position in {3, 4, 5}:
            return token == "<s>"
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
        elif position in {24, 35}:
            return token == "<pad>"

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
            6,
            9,
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
        elif position in {8, 10, 7}:
            return token == "<s>"
        elif position in {20}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_4_output, token):
        key = (attn_0_4_output, token)
        return 4

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_4_outputs, tokens)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {0, 1, 2, 3, 4, 5, 6, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return 32
        return 17

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_3_output, attn_0_6_output):
        key = (attn_0_3_output, attn_0_6_output)
        return 12

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_6_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_7_output, attn_0_3_output):
        key = (attn_0_7_output, attn_0_3_output)
        return 8

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_3_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_7_output, num_attn_0_0_output):
        key = (num_attn_0_7_output, num_attn_0_0_output)
        return 30

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_0_outputs)
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
    def num_mlp_0_2(num_attn_0_0_output, num_attn_0_7_output):
        key = (num_attn_0_0_output, num_attn_0_7_output)
        return 10

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_7_output, num_attn_0_1_output):
        key = (num_attn_0_7_output, num_attn_0_1_output)
        return 13

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 4, 6, 10, 12, 14, 18, 25}:
            return token == "3"
        elif position in {40, 1, 2, 3}:
            return token == "2"
        elif position in {33, 36, 5, 7, 39, 28, 29}:
            return token == "5"
        elif position in {8, 9, 32, 15}:
            return token == "4"
        elif position in {34, 35, 37, 38, 11, 16, 17, 20, 21, 23, 24, 26, 27, 30}:
            return token == "0"
        elif position in {41, 42, 43, 44, 13, 45, 46, 47, 48, 49, 19, 22, 31}:
            return token == ""

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"0"}:
            return position == 2
        elif token in {"1", "3"}:
            return position == 1
        elif token in {"5", "<s>", "2"}:
            return position == 5
        elif token in {"4"}:
            return position == 13

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, positions)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, attn_0_0_output):
        if token in {"0"}:
            return attn_0_0_output == 15
        elif token in {"1"}:
            return attn_0_0_output == 26
        elif token in {"2"}:
            return attn_0_0_output == 23
        elif token in {"3"}:
            return attn_0_0_output == 9
        elif token in {"4"}:
            return attn_0_0_output == 22
        elif token in {"5"}:
            return attn_0_0_output == 29
        elif token in {"<s>"}:
            return attn_0_0_output == 25

    attn_1_2_pattern = select_closest(attn_0_0_outputs, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_6_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0, 1, 2, 8, 11, 16, 20, 23, 24, 27}:
            return token == "3"
        elif position in {32, 3, 35, 36, 7, 43, 12, 13, 25, 28, 29, 31}:
            return token == "4"
        elif position in {34, 4, 5, 18, 26}:
            return token == "0"
        elif position in {6}:
            return token == "1"
        elif position in {
            33,
            37,
            38,
            40,
            9,
            10,
            41,
            42,
            44,
            46,
            15,
            47,
            17,
            48,
            19,
            21,
            22,
            30,
        }:
            return token == ""
        elif position in {14}:
            return token == "2"
        elif position in {49, 45, 39}:
            return token == "<s>"

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_token, k_token):
        if q_token in {"0", "4", "2", "1", "5"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    attn_1_4_pattern = select_closest(tokens, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_token, k_token):
        if q_token in {"0", "<s>"}:
            return k_token == "3"
        elif q_token in {"4", "2", "1", "5", "3"}:
            return k_token == "0"

    attn_1_5_pattern = select_closest(tokens, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "5"
        elif attn_0_2_output in {"1"}:
            return token == "2"
        elif attn_0_2_output in {"2"}:
            return token == "3"
        elif attn_0_2_output in {"5", "<s>", "3"}:
            return token == ""
        elif attn_0_2_output in {"4"}:
            return token == "<s>"

    attn_1_6_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_6_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(token, position):
        if token in {"0", "2"}:
            return position == 6
        elif token in {"1"}:
            return position == 5
        elif token in {"3"}:
            return position == 7
        elif token in {"4", "5"}:
            return position == 1
        elif token in {"<s>"}:
            return position == 41

    attn_1_7_pattern = select_closest(positions, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, positions)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_3_output):
        if position in {0, 16, 17, 7}:
            return attn_0_3_output == "<s>"
        elif position in {8, 1}:
            return attn_0_3_output == "1"
        elif position in {
            2,
            3,
            4,
            5,
            6,
            9,
            11,
            15,
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
            36,
            37,
            38,
            39,
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
            return attn_0_3_output == ""
        elif position in {10, 14}:
            return attn_0_3_output == "3"
        elif position in {12}:
            return attn_0_3_output == "0"
        elif position in {13}:
            return attn_0_3_output == "2"
        elif position in {35, 23}:
            return attn_0_3_output == "<pad>"
        elif position in {40}:
            return attn_0_3_output == "4"

    num_attn_1_0_pattern = select(attn_0_3_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_5_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(token, position):
        if token in {"0"}:
            return position == 36
        elif token in {"1", "4"}:
            return position == 7
        elif token in {"2"}:
            return position == 49
        elif token in {"3"}:
            return position == 31
        elif token in {"5"}:
            return position == 46
        elif token in {"<s>"}:
            return position == 0

    num_attn_1_1_pattern = select(positions, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_7_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, attn_0_2_output):
        if attn_0_1_output in {"0"}:
            return attn_0_2_output == "0"
        elif attn_0_1_output in {"<s>", "4", "2", "1", "5", "3"}:
            return attn_0_2_output == ""

    num_attn_1_2_pattern = select(attn_0_2_outputs, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_5_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_2_output):
        if position in {0, 5, 6, 10, 48}:
            return attn_0_2_output == "3"
        elif position in {
            1,
            2,
            3,
            4,
            8,
            9,
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
            44,
            45,
            46,
            47,
            49,
        }:
            return attn_0_2_output == ""
        elif position in {7}:
            return attn_0_2_output == "2"
        elif position in {17, 25}:
            return attn_0_2_output == "1"
        elif position in {43}:
            return attn_0_2_output == "<pad>"

    num_attn_1_3_pattern = select(attn_0_2_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_4_output, attn_0_1_output):
        if attn_0_4_output in {"0", "4", "2", "1", "5"}:
            return attn_0_1_output == ""
        elif attn_0_4_output in {"<s>", "3"}:
            return attn_0_1_output == "3"

    num_attn_1_4_pattern = select(attn_0_1_outputs, attn_0_4_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_3_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, attn_0_1_output):
        if position in {0, 10, 12, 15}:
            return attn_0_1_output == "0"
        elif position in {
            1,
            2,
            4,
            7,
            8,
            11,
            13,
            14,
            16,
            18,
            19,
            21,
            22,
            23,
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
            return attn_0_1_output == ""
        elif position in {3, 20}:
            return attn_0_1_output == "<pad>"
        elif position in {5, 6}:
            return attn_0_1_output == "4"
        elif position in {9}:
            return attn_0_1_output == "<s>"
        elif position in {24, 17}:
            return attn_0_1_output == "2"

    num_attn_1_5_pattern = select(attn_0_1_outputs, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_6_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_2_output, attn_0_4_output):
        if attn_0_2_output in {"0", "<s>", "4", "2", "3"}:
            return attn_0_4_output == ""
        elif attn_0_2_output in {"1"}:
            return attn_0_4_output == "<pad>"
        elif attn_0_2_output in {"5"}:
            return attn_0_4_output == "5"

    num_attn_1_6_pattern = select(attn_0_4_outputs, attn_0_2_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_1_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_5_output, attn_0_4_output):
        if attn_0_5_output in {"0", "<s>", "4", "1", "5", "3"}:
            return attn_0_4_output == ""
        elif attn_0_5_output in {"2"}:
            return attn_0_4_output == "2"

    num_attn_1_7_pattern = select(attn_0_4_outputs, attn_0_5_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_7_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output):
        key = attn_1_0_output
        return 0

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in attn_1_0_outputs]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_3_output, attn_1_0_output):
        key = (num_mlp_0_3_output, attn_1_0_output)
        return 23

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_3_outputs, attn_1_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_0_output, attn_0_2_output):
        key = (attn_1_0_output, attn_0_2_output)
        return 2

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_0_2_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(position, attn_1_6_output):
        key = (position, attn_1_6_output)
        return 27

    mlp_1_3_outputs = [mlp_1_3(k0, k1) for k0, k1 in zip(positions, attn_1_6_outputs)]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_7_output, num_attn_1_4_output):
        key = (num_attn_0_7_output, num_attn_1_4_output)
        return 20

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_4_output, num_attn_1_0_output):
        key = (num_attn_1_4_output, num_attn_1_0_output)
        return 16

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_3_output, num_attn_1_6_output):
        key = (num_attn_1_3_output, num_attn_1_6_output)
        return 4

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_7_output, num_attn_1_0_output):
        key = (num_attn_1_7_output, num_attn_1_0_output)
        return 25

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "2"
        elif attn_0_4_output in {"1"}:
            return token == "3"
        elif attn_0_4_output in {"2"}:
            return token == "0"
        elif attn_0_4_output in {"<s>", "4", "3"}:
            return token == "1"
        elif attn_0_4_output in {"5"}:
            return token == ""

    attn_2_0_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_token, k_token):
        if q_token in {"0", "4", "2", "1", "3"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_1_pattern = select_closest(tokens, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, tokens)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"0", "4", "2", "5", "3"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, tokens)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 17
        elif q_position in {1}:
            return k_position == 11
        elif q_position in {2}:
            return k_position == 18
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {40, 5}:
            return k_position == 0
        elif q_position in {11, 44, 6}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 38
        elif q_position in {8}:
            return k_position == 35
        elif q_position in {9, 38}:
            return k_position == 29
        elif q_position in {36, 10, 18, 20, 21, 24}:
            return k_position == 22
        elif q_position in {19, 12, 23}:
            return k_position == 5
        elif q_position in {37, 13, 16, 17, 22, 27}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 33
        elif q_position in {15}:
            return k_position == 26
        elif q_position in {32, 35, 41, 25, 29, 30, 31}:
            return k_position == 28
        elif q_position in {33, 26, 28, 34}:
            return k_position == 32
        elif q_position in {39}:
            return k_position == 23
        elif q_position in {42, 43}:
            return k_position == 49
        elif q_position in {45, 47}:
            return k_position == 4
        elif q_position in {46}:
            return k_position == 41
        elif q_position in {48}:
            return k_position == 13
        elif q_position in {49}:
            return k_position == 19

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(position, token):
        if position in {
            0,
            1,
            4,
            5,
            7,
            9,
            12,
            14,
            15,
            17,
            18,
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
            43,
            44,
            46,
            48,
            49,
        }:
            return token == ""
        elif position in {2}:
            return token == "2"
        elif position in {3, 28}:
            return token == "3"
        elif position in {6}:
            return token == "1"
        elif position in {8, 41, 10, 11, 13, 47, 16, 31}:
            return token == "0"
        elif position in {19}:
            return token == "4"
        elif position in {45}:
            return token == "<s>"

    attn_2_4_pattern = select_closest(tokens, positions, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_5_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_1_output, attn_0_5_output):
        if attn_0_1_output in {"0", "<s>", "1", "5", "3"}:
            return attn_0_5_output == "<s>"
        elif attn_0_1_output in {"2"}:
            return attn_0_5_output == "3"
        elif attn_0_1_output in {"4"}:
            return attn_0_5_output == ""

    attn_2_5_pattern = select_closest(attn_0_5_outputs, attn_0_1_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_0_5_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
        if q_token in {"0", "4", "1", "5", "3"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_5_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_attn_1_5_output, k_attn_1_5_output):
        if q_attn_1_5_output in {"0", "<s>", "4", "1", "5"}:
            return k_attn_1_5_output == ""
        elif q_attn_1_5_output in {"2", "3"}:
            return k_attn_1_5_output == "4"

    attn_2_7_pattern = select_closest(attn_1_5_outputs, attn_1_5_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_3_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, attn_0_4_output):
        if position in {0, 1, 2, 3, 37, 40, 41, 42, 13, 45, 48, 49}:
            return attn_0_4_output == ""
        elif position in {4}:
            return attn_0_4_output == "<s>"
        elif position in {
            5,
            6,
            7,
            8,
            10,
            12,
            15,
            17,
            18,
            21,
            22,
            23,
            25,
            27,
            28,
            29,
            32,
            33,
            34,
            35,
            36,
            43,
            44,
            46,
            47,
        }:
            return attn_0_4_output == "0"
        elif position in {9}:
            return attn_0_4_output == "5"
        elif position in {19, 24, 11, 14}:
            return attn_0_4_output == "3"
        elif position in {38, 16, 20, 26, 30, 31}:
            return attn_0_4_output == "4"
        elif position in {39}:
            return attn_0_4_output == "2"

    num_attn_2_0_pattern = select(attn_0_4_outputs, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_1_0_output):
        if position in {0, 40, 42, 43, 46, 49}:
            return attn_1_0_output == "4"
        elif position in {1, 47}:
            return attn_1_0_output == "3"
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
            13,
            14,
            15,
            16,
            17,
            18,
            19,
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
            48,
        }:
            return attn_1_0_output == ""
        elif position in {35, 20, 21}:
            return attn_1_0_output == "<pad>"
        elif position in {41, 44}:
            return attn_1_0_output == "0"
        elif position in {45}:
            return attn_1_0_output == "<s>"

    num_attn_2_1_pattern = select(attn_1_0_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_2_output, attn_0_1_output):
        if attn_0_2_output in {"0", "<s>", "4", "1", "3"}:
            return attn_0_1_output == "5"
        elif attn_0_2_output in {"2"}:
            return attn_0_1_output == ""
        elif attn_0_2_output in {"5"}:
            return attn_0_1_output == "1"

    num_attn_2_2_pattern = select(attn_0_1_outputs, attn_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(token, attn_1_2_output):
        if token in {"0", "<s>", "4", "2", "5", "3"}:
            return attn_1_2_output == ""
        elif token in {"1"}:
            return attn_1_2_output == "1"

    num_attn_2_3_pattern = select(attn_1_2_outputs, tokens, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_6_output, attn_0_4_output):
        if attn_0_6_output in {"0", "2", "1", "5", "3"}:
            return attn_0_4_output == ""
        elif attn_0_6_output in {"4"}:
            return attn_0_4_output == "4"
        elif attn_0_6_output in {"<s>"}:
            return attn_0_4_output == "<pad>"

    num_attn_2_4_pattern = select(attn_0_4_outputs, attn_0_6_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_6_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(token, attn_1_0_output):
        if token in {"0", "2", "1", "5", "3"}:
            return attn_1_0_output == ""
        elif token in {"4"}:
            return attn_1_0_output == "4"
        elif token in {"<s>"}:
            return attn_1_0_output == "<pad>"

    num_attn_2_5_pattern = select(attn_1_0_outputs, tokens, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_6_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(token, position):
        if token in {"0", "3"}:
            return position == 43
        elif token in {"1"}:
            return position == 7
        elif token in {"2"}:
            return position == 46
        elif token in {"<s>", "4"}:
            return position == 45
        elif token in {"5"}:
            return position == 6

    num_attn_2_6_pattern = select(positions, tokens, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_1_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(token, attn_1_1_output):
        if token in {"0"}:
            return attn_1_1_output == 9
        elif token in {"1"}:
            return attn_1_1_output == 33
        elif token in {"2"}:
            return attn_1_1_output == 43
        elif token in {"3"}:
            return attn_1_1_output == 41
        elif token in {"4"}:
            return attn_1_1_output == 49
        elif token in {"5"}:
            return attn_1_1_output == 19
        elif token in {"<s>"}:
            return attn_1_1_output == 47

    num_attn_2_7_pattern = select(attn_1_1_outputs, tokens, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_1_3_output, attn_1_1_output):
        key = (num_mlp_1_3_output, attn_1_1_output)
        if key in {
            (11, 12),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (13, 11),
            (13, 12),
            (13, 13),
            (13, 15),
            (13, 16),
            (13, 17),
            (13, 20),
            (13, 21),
            (13, 22),
            (13, 23),
            (13, 24),
            (13, 25),
            (13, 26),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 31),
            (13, 32),
            (13, 33),
            (13, 34),
            (13, 35),
            (13, 36),
            (13, 37),
            (13, 38),
            (13, 39),
            (13, 41),
            (13, 43),
            (13, 44),
            (13, 45),
            (13, 46),
            (13, 47),
            (13, 48),
            (13, 49),
            (31, 21),
            (42, 0),
            (42, 6),
            (42, 8),
            (42, 12),
            (42, 37),
            (42, 41),
        }:
            return 5
        elif key in {(31, 44)}:
            return 4
        return 25

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_1_3_outputs, attn_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_5_output, attn_2_5_output):
        key = (attn_1_5_output, attn_2_5_output)
        return 26

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_5_outputs, attn_2_5_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_7_output, position):
        key = (attn_2_7_output, position)
        if key in {
            ("0", 5),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("0", 14),
            ("0", 15),
            ("0", 16),
            ("0", 17),
            ("0", 18),
            ("0", 19),
            ("0", 20),
            ("0", 23),
            ("0", 25),
            ("0", 26),
            ("0", 28),
            ("0", 30),
            ("0", 31),
            ("0", 35),
            ("1", 5),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 12),
            ("1", 13),
            ("1", 14),
            ("1", 15),
            ("1", 16),
            ("1", 17),
            ("1", 18),
            ("1", 19),
            ("1", 20),
            ("1", 23),
            ("1", 25),
            ("1", 26),
            ("1", 28),
            ("1", 29),
            ("1", 30),
            ("1", 31),
            ("1", 35),
            ("2", 0),
            ("2", 5),
            ("2", 9),
            ("2", 10),
            ("2", 11),
            ("2", 12),
            ("2", 13),
            ("2", 14),
            ("2", 15),
            ("2", 16),
            ("2", 17),
            ("2", 18),
            ("2", 19),
            ("2", 20),
            ("2", 23),
            ("2", 25),
            ("2", 26),
            ("2", 28),
            ("2", 31),
            ("4", 9),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 14),
            ("4", 16),
            ("4", 18),
            ("4", 23),
            ("5", 5),
            ("5", 9),
            ("5", 10),
            ("5", 11),
            ("5", 12),
            ("5", 13),
            ("5", 14),
            ("5", 15),
            ("5", 16),
            ("5", 17),
            ("5", 18),
            ("5", 19),
            ("5", 20),
            ("5", 23),
            ("5", 25),
            ("5", 26),
            ("5", 28),
            ("<s>", 9),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 16),
            ("<s>", 18),
            ("<s>", 23),
        }:
            return 32
        elif key in {
            ("0", 0),
            ("0", 1),
            ("1", 0),
            ("1", 1),
            ("2", 1),
            ("5", 0),
            ("5", 1),
            ("<s>", 1),
        }:
            return 14
        return 22

    mlp_2_2_outputs = [mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_7_outputs, positions)]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_0_2_output, mlp_1_2_output):
        key = (num_mlp_0_2_output, mlp_1_2_output)
        return 41

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_0_2_outputs, mlp_1_2_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_2_2_output):
        key = (num_attn_2_1_output, num_attn_2_2_output)
        return 19

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_7_output, num_attn_2_2_output):
        key = (num_attn_1_7_output, num_attn_2_2_output)
        return 19

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_0_output, num_attn_1_7_output):
        key = (num_attn_2_0_output, num_attn_1_7_output)
        if key in {
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 48),
            (0, 49),
            (0, 50),
            (0, 51),
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (0, 60),
            (0, 61),
            (0, 62),
            (0, 63),
            (0, 64),
            (0, 65),
            (0, 66),
            (0, 67),
            (0, 68),
            (0, 69),
            (0, 70),
            (0, 71),
            (0, 72),
            (0, 73),
            (0, 74),
            (0, 75),
            (0, 76),
            (0, 77),
            (0, 78),
            (0, 79),
            (0, 80),
            (0, 81),
            (0, 82),
            (0, 83),
            (0, 84),
            (0, 85),
            (0, 86),
            (0, 87),
            (0, 88),
            (0, 89),
            (0, 90),
            (0, 91),
            (0, 92),
            (0, 93),
            (0, 94),
            (0, 95),
            (0, 96),
            (0, 97),
            (0, 98),
            (0, 99),
            (0, 100),
            (0, 101),
            (0, 102),
            (0, 103),
            (0, 104),
            (0, 105),
            (0, 106),
            (0, 107),
            (0, 108),
            (0, 109),
            (0, 110),
            (0, 111),
            (0, 112),
            (0, 113),
            (0, 114),
            (0, 115),
            (0, 116),
            (0, 117),
            (0, 118),
            (0, 119),
            (0, 120),
            (0, 121),
            (0, 122),
            (0, 123),
            (0, 124),
            (0, 125),
            (0, 126),
            (0, 127),
            (0, 128),
            (0, 129),
            (0, 130),
            (0, 131),
            (0, 132),
            (0, 133),
            (0, 134),
            (0, 135),
            (0, 136),
            (0, 137),
            (0, 138),
            (0, 139),
            (0, 140),
            (0, 141),
            (0, 142),
            (0, 143),
            (0, 144),
            (0, 145),
            (0, 146),
            (0, 147),
            (0, 148),
            (0, 149),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (1, 48),
            (1, 49),
            (1, 50),
            (1, 51),
            (1, 52),
            (1, 53),
            (1, 54),
            (1, 55),
            (1, 56),
            (1, 57),
            (1, 58),
            (1, 59),
            (1, 60),
            (1, 61),
            (1, 62),
            (1, 63),
            (1, 64),
            (1, 65),
            (1, 66),
            (1, 67),
            (1, 68),
            (1, 69),
            (1, 70),
            (1, 71),
            (1, 72),
            (1, 73),
            (1, 74),
            (1, 75),
            (1, 76),
            (1, 77),
            (1, 78),
            (1, 79),
            (1, 80),
            (1, 81),
            (1, 82),
            (1, 83),
            (1, 84),
            (1, 85),
            (1, 86),
            (1, 87),
            (1, 88),
            (1, 89),
            (1, 90),
            (1, 91),
            (1, 92),
            (1, 93),
            (1, 94),
            (1, 95),
            (1, 96),
            (1, 97),
            (1, 98),
            (1, 99),
            (1, 100),
            (1, 101),
            (1, 102),
            (1, 103),
            (1, 104),
            (1, 105),
            (1, 106),
            (1, 107),
            (1, 108),
            (1, 109),
            (1, 110),
            (1, 111),
            (1, 112),
            (1, 113),
            (1, 114),
            (1, 115),
            (1, 116),
            (1, 117),
            (1, 118),
            (1, 119),
            (1, 120),
            (1, 121),
            (1, 122),
            (1, 123),
            (1, 124),
            (1, 125),
            (1, 126),
            (1, 127),
            (1, 128),
            (1, 129),
            (1, 130),
            (1, 131),
            (1, 132),
            (1, 133),
            (1, 134),
            (1, 135),
            (1, 136),
            (1, 137),
            (1, 138),
            (1, 139),
            (1, 140),
            (1, 141),
            (1, 142),
            (1, 143),
            (1, 144),
            (1, 145),
            (1, 146),
            (1, 147),
            (1, 148),
            (1, 149),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (2, 48),
            (2, 49),
            (2, 50),
            (2, 51),
            (2, 52),
            (2, 53),
            (2, 54),
            (2, 55),
            (2, 56),
            (2, 57),
            (2, 58),
            (2, 59),
            (2, 60),
            (2, 61),
            (2, 62),
            (2, 63),
            (2, 64),
            (2, 65),
            (2, 66),
            (2, 67),
            (2, 68),
            (2, 69),
            (2, 70),
            (2, 71),
            (2, 72),
            (2, 73),
            (2, 74),
            (2, 75),
            (2, 76),
            (2, 77),
            (2, 78),
            (2, 79),
            (2, 80),
            (2, 81),
            (2, 82),
            (2, 83),
            (2, 84),
            (2, 85),
            (2, 86),
            (2, 87),
            (2, 88),
            (2, 89),
            (2, 90),
            (2, 91),
            (2, 92),
            (2, 93),
            (2, 94),
            (2, 95),
            (2, 96),
            (2, 97),
            (2, 98),
            (2, 99),
            (2, 100),
            (2, 101),
            (2, 102),
            (2, 103),
            (2, 104),
            (2, 105),
            (2, 106),
            (2, 107),
            (2, 108),
            (2, 109),
            (2, 110),
            (2, 111),
            (2, 112),
            (2, 113),
            (2, 114),
            (2, 115),
            (2, 116),
            (2, 117),
            (2, 118),
            (2, 119),
            (2, 120),
            (2, 121),
            (2, 122),
            (2, 123),
            (2, 124),
            (2, 125),
            (2, 126),
            (2, 127),
            (2, 128),
            (2, 129),
            (2, 130),
            (2, 131),
            (2, 132),
            (2, 133),
            (2, 134),
            (2, 135),
            (2, 136),
            (2, 137),
            (2, 138),
            (2, 139),
            (2, 140),
            (2, 141),
            (2, 142),
            (2, 143),
            (2, 144),
            (2, 145),
            (2, 146),
            (2, 147),
            (2, 148),
            (2, 149),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 34),
            (3, 35),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (3, 40),
            (3, 41),
            (3, 42),
            (3, 43),
            (3, 44),
            (3, 45),
            (3, 46),
            (3, 47),
            (3, 48),
            (3, 49),
            (3, 50),
            (3, 51),
            (3, 52),
            (3, 53),
            (3, 54),
            (3, 55),
            (3, 56),
            (3, 57),
            (3, 58),
            (3, 59),
            (3, 60),
            (3, 61),
            (3, 62),
            (3, 63),
            (3, 64),
            (3, 65),
            (3, 66),
            (3, 67),
            (3, 68),
            (3, 69),
            (3, 70),
            (3, 71),
            (3, 72),
            (3, 73),
            (3, 74),
            (3, 75),
            (3, 76),
            (3, 77),
            (3, 78),
            (3, 79),
            (3, 80),
            (3, 81),
            (3, 82),
            (3, 83),
            (3, 84),
            (3, 85),
            (3, 86),
            (3, 87),
            (3, 88),
            (3, 89),
            (3, 90),
            (3, 91),
            (3, 92),
            (3, 93),
            (3, 94),
            (3, 95),
            (3, 96),
            (3, 97),
            (3, 98),
            (3, 99),
            (3, 100),
            (3, 101),
            (3, 102),
            (3, 103),
            (3, 104),
            (3, 105),
            (3, 106),
            (3, 107),
            (3, 108),
            (3, 109),
            (3, 110),
            (3, 111),
            (3, 112),
            (3, 113),
            (3, 114),
            (3, 115),
            (3, 116),
            (3, 117),
            (3, 118),
            (3, 119),
            (3, 120),
            (3, 121),
            (3, 122),
            (3, 123),
            (3, 124),
            (3, 125),
            (3, 126),
            (3, 127),
            (3, 128),
            (3, 129),
            (3, 130),
            (3, 131),
            (3, 132),
            (3, 133),
            (3, 134),
            (3, 135),
            (3, 136),
            (3, 137),
            (3, 138),
            (3, 139),
            (3, 140),
            (3, 141),
            (3, 142),
            (3, 143),
            (3, 144),
            (3, 145),
            (3, 146),
            (3, 147),
            (3, 148),
            (3, 149),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (4, 40),
            (4, 41),
            (4, 42),
            (4, 43),
            (4, 44),
            (4, 45),
            (4, 46),
            (4, 47),
            (4, 48),
            (4, 49),
            (4, 50),
            (4, 51),
            (4, 52),
            (4, 53),
            (4, 54),
            (4, 55),
            (4, 56),
            (4, 57),
            (4, 58),
            (4, 59),
            (4, 60),
            (4, 61),
            (4, 62),
            (4, 63),
            (4, 64),
            (4, 65),
            (4, 66),
            (4, 67),
            (4, 68),
            (4, 69),
            (4, 70),
            (4, 71),
            (4, 72),
            (4, 73),
            (4, 74),
            (4, 75),
            (4, 76),
            (4, 77),
            (4, 78),
            (4, 79),
            (4, 80),
            (4, 81),
            (4, 82),
            (4, 83),
            (4, 84),
            (4, 85),
            (4, 86),
            (4, 87),
            (4, 88),
            (4, 89),
            (4, 90),
            (4, 91),
            (4, 92),
            (4, 93),
            (4, 94),
            (4, 95),
            (4, 96),
            (4, 97),
            (4, 98),
            (4, 99),
            (4, 100),
            (4, 101),
            (4, 102),
            (4, 103),
            (4, 104),
            (4, 105),
            (4, 106),
            (4, 107),
            (4, 108),
            (4, 109),
            (4, 110),
            (4, 111),
            (4, 112),
            (4, 113),
            (4, 114),
            (4, 115),
            (4, 116),
            (4, 117),
            (4, 118),
            (4, 119),
            (4, 120),
            (4, 121),
            (4, 122),
            (4, 123),
            (4, 124),
            (4, 125),
            (4, 126),
            (4, 127),
            (4, 128),
            (4, 129),
            (4, 130),
            (4, 131),
            (4, 132),
            (4, 133),
            (4, 134),
            (4, 135),
            (4, 136),
            (4, 137),
            (4, 138),
            (4, 139),
            (4, 140),
            (4, 141),
            (4, 142),
            (4, 143),
            (4, 144),
            (4, 145),
            (4, 146),
            (4, 147),
            (4, 148),
            (4, 149),
            (5, 23),
            (5, 24),
            (5, 25),
            (5, 26),
            (5, 27),
            (5, 28),
            (5, 29),
            (5, 30),
            (5, 31),
            (5, 32),
            (5, 33),
            (5, 34),
            (5, 35),
            (5, 36),
            (5, 37),
            (5, 38),
            (5, 39),
            (5, 40),
            (5, 41),
            (5, 42),
            (5, 43),
            (5, 44),
            (5, 45),
            (5, 46),
            (5, 47),
            (5, 48),
            (5, 49),
            (5, 50),
            (5, 51),
            (5, 52),
            (5, 53),
            (5, 54),
            (5, 55),
            (5, 56),
            (5, 57),
            (5, 58),
            (5, 59),
            (5, 60),
            (5, 61),
            (5, 62),
            (5, 63),
            (5, 64),
            (5, 65),
            (5, 66),
            (5, 67),
            (5, 68),
            (5, 69),
            (5, 70),
            (5, 71),
            (5, 72),
            (5, 73),
            (5, 74),
            (5, 75),
            (5, 76),
            (5, 77),
            (5, 78),
            (5, 79),
            (5, 80),
            (5, 81),
            (5, 82),
            (5, 83),
            (5, 84),
            (5, 85),
            (5, 86),
            (5, 87),
            (5, 88),
            (5, 89),
            (5, 90),
            (5, 91),
            (5, 92),
            (5, 93),
            (5, 94),
            (5, 95),
            (5, 96),
            (5, 97),
            (5, 98),
            (5, 99),
            (5, 100),
            (5, 101),
            (5, 102),
            (5, 103),
            (5, 104),
            (5, 105),
            (5, 106),
            (5, 107),
            (5, 108),
            (5, 109),
            (5, 110),
            (5, 111),
            (5, 112),
            (5, 113),
            (5, 114),
            (5, 115),
            (5, 116),
            (5, 117),
            (5, 118),
            (5, 119),
            (5, 120),
            (5, 121),
            (5, 122),
            (5, 123),
            (5, 124),
            (5, 125),
            (5, 126),
            (5, 127),
            (5, 128),
            (5, 129),
            (5, 130),
            (5, 131),
            (5, 132),
            (5, 133),
            (5, 134),
            (5, 135),
            (5, 136),
            (5, 137),
            (5, 138),
            (5, 139),
            (5, 140),
            (5, 141),
            (5, 142),
            (5, 143),
            (5, 144),
            (5, 145),
            (5, 146),
            (5, 147),
            (5, 148),
            (5, 149),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (6, 40),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (6, 48),
            (6, 49),
            (6, 50),
            (6, 51),
            (6, 52),
            (6, 53),
            (6, 54),
            (6, 55),
            (6, 56),
            (6, 57),
            (6, 58),
            (6, 59),
            (6, 60),
            (6, 61),
            (6, 62),
            (6, 63),
            (6, 64),
            (6, 65),
            (6, 66),
            (6, 67),
            (6, 68),
            (6, 69),
            (6, 70),
            (6, 71),
            (6, 72),
            (6, 73),
            (6, 74),
            (6, 75),
            (6, 76),
            (6, 77),
            (6, 78),
            (6, 79),
            (6, 80),
            (6, 81),
            (6, 82),
            (6, 83),
            (6, 84),
            (6, 85),
            (6, 86),
            (6, 87),
            (6, 88),
            (6, 89),
            (6, 90),
            (6, 91),
            (6, 92),
            (6, 93),
            (6, 94),
            (6, 95),
            (6, 96),
            (6, 97),
            (6, 98),
            (6, 99),
            (6, 100),
            (6, 101),
            (6, 102),
            (6, 103),
            (6, 104),
            (6, 105),
            (6, 106),
            (6, 107),
            (6, 108),
            (6, 109),
            (6, 110),
            (6, 111),
            (6, 112),
            (6, 113),
            (6, 114),
            (6, 115),
            (6, 116),
            (6, 117),
            (6, 118),
            (6, 119),
            (6, 120),
            (6, 121),
            (6, 122),
            (6, 123),
            (6, 124),
            (6, 125),
            (6, 126),
            (6, 127),
            (6, 128),
            (6, 129),
            (6, 130),
            (6, 131),
            (6, 132),
            (6, 133),
            (6, 134),
            (6, 135),
            (6, 136),
            (6, 137),
            (6, 138),
            (6, 139),
            (6, 140),
            (6, 141),
            (6, 142),
            (6, 143),
            (6, 144),
            (6, 145),
            (6, 146),
            (6, 147),
            (6, 148),
            (6, 149),
            (7, 31),
            (7, 32),
            (7, 33),
            (7, 34),
            (7, 35),
            (7, 36),
            (7, 37),
            (7, 38),
            (7, 39),
            (7, 40),
            (7, 41),
            (7, 42),
            (7, 43),
            (7, 44),
            (7, 45),
            (7, 46),
            (7, 47),
            (7, 48),
            (7, 49),
            (7, 50),
            (7, 51),
            (7, 52),
            (7, 53),
            (7, 54),
            (7, 55),
            (7, 56),
            (7, 57),
            (7, 58),
            (7, 59),
            (7, 60),
            (7, 61),
            (7, 62),
            (7, 63),
            (7, 64),
            (7, 65),
            (7, 66),
            (7, 67),
            (7, 68),
            (7, 69),
            (7, 70),
            (7, 71),
            (7, 72),
            (7, 73),
            (7, 74),
            (7, 75),
            (7, 76),
            (7, 77),
            (7, 78),
            (7, 79),
            (7, 80),
            (7, 81),
            (7, 82),
            (7, 83),
            (7, 84),
            (7, 85),
            (7, 86),
            (7, 87),
            (7, 88),
            (7, 89),
            (7, 90),
            (7, 91),
            (7, 92),
            (7, 93),
            (7, 94),
            (7, 95),
            (7, 96),
            (7, 97),
            (7, 98),
            (7, 99),
            (7, 100),
            (7, 101),
            (7, 102),
            (7, 103),
            (7, 104),
            (7, 105),
            (7, 106),
            (7, 107),
            (7, 108),
            (7, 109),
            (7, 110),
            (7, 111),
            (7, 112),
            (7, 113),
            (7, 114),
            (7, 115),
            (7, 116),
            (7, 117),
            (7, 118),
            (7, 119),
            (7, 120),
            (7, 121),
            (7, 122),
            (7, 123),
            (7, 124),
            (7, 125),
            (7, 126),
            (7, 127),
            (7, 128),
            (7, 129),
            (7, 130),
            (7, 131),
            (7, 132),
            (7, 133),
            (7, 134),
            (7, 135),
            (7, 136),
            (7, 137),
            (7, 138),
            (7, 139),
            (7, 140),
            (7, 141),
            (7, 142),
            (7, 143),
            (7, 144),
            (7, 145),
            (7, 146),
            (7, 147),
            (7, 148),
            (7, 149),
            (8, 35),
            (8, 36),
            (8, 37),
            (8, 38),
            (8, 39),
            (8, 40),
            (8, 41),
            (8, 42),
            (8, 43),
            (8, 44),
            (8, 45),
            (8, 46),
            (8, 47),
            (8, 48),
            (8, 49),
            (8, 50),
            (8, 51),
            (8, 52),
            (8, 53),
            (8, 54),
            (8, 55),
            (8, 56),
            (8, 57),
            (8, 58),
            (8, 59),
            (8, 60),
            (8, 61),
            (8, 62),
            (8, 63),
            (8, 64),
            (8, 65),
            (8, 66),
            (8, 67),
            (8, 68),
            (8, 69),
            (8, 70),
            (8, 71),
            (8, 72),
            (8, 73),
            (8, 74),
            (8, 75),
            (8, 76),
            (8, 77),
            (8, 78),
            (8, 79),
            (8, 80),
            (8, 81),
            (8, 82),
            (8, 83),
            (8, 84),
            (8, 85),
            (8, 86),
            (8, 87),
            (8, 88),
            (8, 89),
            (8, 90),
            (8, 91),
            (8, 92),
            (8, 93),
            (8, 94),
            (8, 95),
            (8, 96),
            (8, 97),
            (8, 98),
            (8, 99),
            (8, 100),
            (8, 101),
            (8, 102),
            (8, 103),
            (8, 104),
            (8, 105),
            (8, 106),
            (8, 107),
            (8, 108),
            (8, 109),
            (8, 110),
            (8, 111),
            (8, 112),
            (8, 113),
            (8, 114),
            (8, 115),
            (8, 116),
            (8, 117),
            (8, 118),
            (8, 119),
            (8, 120),
            (8, 121),
            (8, 122),
            (8, 123),
            (8, 124),
            (8, 125),
            (8, 126),
            (8, 127),
            (8, 128),
            (8, 129),
            (8, 130),
            (8, 131),
            (8, 132),
            (8, 133),
            (8, 134),
            (8, 135),
            (8, 136),
            (8, 137),
            (8, 138),
            (8, 139),
            (8, 140),
            (8, 141),
            (8, 142),
            (8, 143),
            (8, 144),
            (8, 145),
            (8, 146),
            (8, 147),
            (8, 148),
            (8, 149),
            (9, 39),
            (9, 40),
            (9, 41),
            (9, 42),
            (9, 43),
            (9, 44),
            (9, 45),
            (9, 46),
            (9, 47),
            (9, 48),
            (9, 49),
            (9, 50),
            (9, 51),
            (9, 52),
            (9, 53),
            (9, 54),
            (9, 55),
            (9, 56),
            (9, 57),
            (9, 58),
            (9, 59),
            (9, 60),
            (9, 61),
            (9, 62),
            (9, 63),
            (9, 64),
            (9, 65),
            (9, 66),
            (9, 67),
            (9, 68),
            (9, 69),
            (9, 70),
            (9, 71),
            (9, 72),
            (9, 73),
            (9, 74),
            (9, 75),
            (9, 76),
            (9, 77),
            (9, 78),
            (9, 79),
            (9, 80),
            (9, 81),
            (9, 82),
            (9, 83),
            (9, 84),
            (9, 85),
            (9, 86),
            (9, 87),
            (9, 88),
            (9, 89),
            (9, 90),
            (9, 91),
            (9, 92),
            (9, 93),
            (9, 94),
            (9, 95),
            (9, 96),
            (9, 97),
            (9, 98),
            (9, 99),
            (9, 100),
            (9, 101),
            (9, 102),
            (9, 103),
            (9, 104),
            (9, 105),
            (9, 106),
            (9, 107),
            (9, 108),
            (9, 109),
            (9, 110),
            (9, 111),
            (9, 112),
            (9, 113),
            (9, 114),
            (9, 115),
            (9, 116),
            (9, 117),
            (9, 118),
            (9, 119),
            (9, 120),
            (9, 121),
            (9, 122),
            (9, 123),
            (9, 124),
            (9, 125),
            (9, 126),
            (9, 127),
            (9, 128),
            (9, 129),
            (9, 130),
            (9, 131),
            (9, 132),
            (9, 133),
            (9, 134),
            (9, 135),
            (9, 136),
            (9, 137),
            (9, 138),
            (9, 139),
            (9, 140),
            (9, 141),
            (9, 142),
            (9, 143),
            (9, 144),
            (9, 145),
            (9, 146),
            (9, 147),
            (9, 148),
            (9, 149),
            (10, 43),
            (10, 44),
            (10, 45),
            (10, 46),
            (10, 47),
            (10, 48),
            (10, 49),
            (10, 50),
            (10, 51),
            (10, 52),
            (10, 53),
            (10, 54),
            (10, 55),
            (10, 56),
            (10, 57),
            (10, 58),
            (10, 59),
            (10, 60),
            (10, 61),
            (10, 62),
            (10, 63),
            (10, 64),
            (10, 65),
            (10, 66),
            (10, 67),
            (10, 68),
            (10, 69),
            (10, 70),
            (10, 71),
            (10, 72),
            (10, 73),
            (10, 74),
            (10, 75),
            (10, 76),
            (10, 77),
            (10, 78),
            (10, 79),
            (10, 80),
            (10, 81),
            (10, 82),
            (10, 83),
            (10, 84),
            (10, 85),
            (10, 86),
            (10, 87),
            (10, 88),
            (10, 89),
            (10, 90),
            (10, 91),
            (10, 92),
            (10, 93),
            (10, 94),
            (10, 95),
            (10, 96),
            (10, 97),
            (10, 98),
            (10, 99),
            (10, 100),
            (10, 101),
            (10, 102),
            (10, 103),
            (10, 104),
            (10, 105),
            (10, 106),
            (10, 107),
            (10, 108),
            (10, 109),
            (10, 110),
            (10, 111),
            (10, 112),
            (10, 113),
            (10, 114),
            (10, 115),
            (10, 116),
            (10, 117),
            (10, 118),
            (10, 119),
            (10, 120),
            (10, 121),
            (10, 122),
            (10, 123),
            (10, 124),
            (10, 125),
            (10, 126),
            (10, 127),
            (10, 128),
            (10, 129),
            (10, 130),
            (10, 131),
            (10, 132),
            (10, 133),
            (10, 134),
            (10, 135),
            (10, 136),
            (10, 137),
            (10, 138),
            (10, 139),
            (10, 140),
            (10, 141),
            (10, 142),
            (10, 143),
            (10, 144),
            (10, 145),
            (10, 146),
            (10, 147),
            (10, 148),
            (10, 149),
            (11, 47),
            (11, 48),
            (11, 49),
            (11, 50),
            (11, 51),
            (11, 52),
            (11, 53),
            (11, 54),
            (11, 55),
            (11, 56),
            (11, 57),
            (11, 58),
            (11, 59),
            (11, 60),
            (11, 61),
            (11, 62),
            (11, 63),
            (11, 64),
            (11, 65),
            (11, 66),
            (11, 67),
            (11, 68),
            (11, 69),
            (11, 70),
            (11, 71),
            (11, 72),
            (11, 73),
            (11, 74),
            (11, 75),
            (11, 76),
            (11, 77),
            (11, 78),
            (11, 79),
            (11, 80),
            (11, 81),
            (11, 82),
            (11, 83),
            (11, 84),
            (11, 85),
            (11, 86),
            (11, 87),
            (11, 88),
            (11, 89),
            (11, 90),
            (11, 91),
            (11, 92),
            (11, 93),
            (11, 94),
            (11, 95),
            (11, 96),
            (11, 97),
            (11, 98),
            (11, 99),
            (11, 100),
            (11, 101),
            (11, 102),
            (11, 103),
            (11, 104),
            (11, 105),
            (11, 106),
            (11, 107),
            (11, 108),
            (11, 109),
            (11, 110),
            (11, 111),
            (11, 112),
            (11, 113),
            (11, 114),
            (11, 115),
            (11, 116),
            (11, 117),
            (11, 118),
            (11, 119),
            (11, 120),
            (11, 121),
            (11, 122),
            (11, 123),
            (11, 124),
            (11, 125),
            (11, 126),
            (11, 127),
            (11, 128),
            (11, 129),
            (11, 130),
            (11, 131),
            (11, 132),
            (11, 133),
            (11, 134),
            (11, 135),
            (11, 136),
            (11, 137),
            (11, 138),
            (11, 139),
            (11, 140),
            (11, 141),
            (11, 142),
            (11, 143),
            (11, 144),
            (11, 145),
            (11, 146),
            (11, 147),
            (11, 148),
            (11, 149),
            (12, 51),
            (12, 52),
            (12, 53),
            (12, 54),
            (12, 55),
            (12, 56),
            (12, 57),
            (12, 58),
            (12, 59),
            (12, 60),
            (12, 61),
            (12, 62),
            (12, 63),
            (12, 64),
            (12, 65),
            (12, 66),
            (12, 67),
            (12, 68),
            (12, 69),
            (12, 70),
            (12, 71),
            (12, 72),
            (12, 73),
            (12, 74),
            (12, 75),
            (12, 76),
            (12, 77),
            (12, 78),
            (12, 79),
            (12, 80),
            (12, 81),
            (12, 82),
            (12, 83),
            (12, 84),
            (12, 85),
            (12, 86),
            (12, 87),
            (12, 88),
            (12, 89),
            (12, 90),
            (12, 91),
            (12, 92),
            (12, 93),
            (12, 94),
            (12, 95),
            (12, 96),
            (12, 97),
            (12, 98),
            (12, 99),
            (12, 100),
            (12, 101),
            (12, 102),
            (12, 103),
            (12, 104),
            (12, 105),
            (12, 106),
            (12, 107),
            (12, 108),
            (12, 109),
            (12, 110),
            (12, 111),
            (12, 112),
            (12, 113),
            (12, 114),
            (12, 115),
            (12, 116),
            (12, 117),
            (12, 118),
            (12, 119),
            (12, 120),
            (12, 121),
            (12, 122),
            (12, 123),
            (12, 124),
            (12, 125),
            (12, 126),
            (12, 127),
            (12, 128),
            (12, 129),
            (12, 130),
            (12, 131),
            (12, 132),
            (12, 133),
            (12, 134),
            (12, 135),
            (12, 136),
            (12, 137),
            (12, 138),
            (12, 139),
            (12, 140),
            (12, 141),
            (12, 142),
            (12, 143),
            (12, 144),
            (12, 145),
            (12, 146),
            (12, 147),
            (12, 148),
            (12, 149),
            (13, 55),
            (13, 56),
            (13, 57),
            (13, 58),
            (13, 59),
            (13, 60),
            (13, 61),
            (13, 62),
            (13, 63),
            (13, 64),
            (13, 65),
            (13, 66),
            (13, 67),
            (13, 68),
            (13, 69),
            (13, 70),
            (13, 71),
            (13, 72),
            (13, 73),
            (13, 74),
            (13, 75),
            (13, 76),
            (13, 77),
            (13, 78),
            (13, 79),
            (13, 80),
            (13, 81),
            (13, 82),
            (13, 83),
            (13, 84),
            (13, 85),
            (13, 86),
            (13, 87),
            (13, 88),
            (13, 89),
            (13, 90),
            (13, 91),
            (13, 92),
            (13, 93),
            (13, 94),
            (13, 95),
            (13, 96),
            (13, 97),
            (13, 98),
            (13, 99),
            (13, 100),
            (13, 101),
            (13, 102),
            (13, 103),
            (13, 104),
            (13, 105),
            (13, 106),
            (13, 107),
            (13, 108),
            (13, 109),
            (13, 110),
            (13, 111),
            (13, 112),
            (13, 113),
            (13, 114),
            (13, 115),
            (13, 116),
            (13, 117),
            (13, 118),
            (13, 119),
            (13, 120),
            (13, 121),
            (13, 122),
            (13, 123),
            (13, 124),
            (13, 125),
            (13, 126),
            (13, 127),
            (13, 128),
            (13, 129),
            (13, 130),
            (13, 131),
            (13, 132),
            (13, 133),
            (13, 134),
            (13, 135),
            (13, 136),
            (13, 137),
            (13, 138),
            (13, 139),
            (13, 140),
            (13, 141),
            (13, 142),
            (13, 143),
            (13, 144),
            (13, 145),
            (13, 146),
            (13, 147),
            (13, 148),
            (13, 149),
            (14, 59),
            (14, 60),
            (14, 61),
            (14, 62),
            (14, 63),
            (14, 64),
            (14, 65),
            (14, 66),
            (14, 67),
            (14, 68),
            (14, 69),
            (14, 70),
            (14, 71),
            (14, 72),
            (14, 73),
            (14, 74),
            (14, 75),
            (14, 76),
            (14, 77),
            (14, 78),
            (14, 79),
            (14, 80),
            (14, 81),
            (14, 82),
            (14, 83),
            (14, 84),
            (14, 85),
            (14, 86),
            (14, 87),
            (14, 88),
            (14, 89),
            (14, 90),
            (14, 91),
            (14, 92),
            (14, 93),
            (14, 94),
            (14, 95),
            (14, 96),
            (14, 97),
            (14, 98),
            (14, 99),
            (14, 100),
            (14, 101),
            (14, 102),
            (14, 103),
            (14, 104),
            (14, 105),
            (14, 106),
            (14, 107),
            (14, 108),
            (14, 109),
            (14, 110),
            (14, 111),
            (14, 112),
            (14, 113),
            (14, 114),
            (14, 115),
            (14, 116),
            (14, 117),
            (14, 118),
            (14, 119),
            (14, 120),
            (14, 121),
            (14, 122),
            (14, 123),
            (14, 124),
            (14, 125),
            (14, 126),
            (14, 127),
            (14, 128),
            (14, 129),
            (14, 130),
            (14, 131),
            (14, 132),
            (14, 133),
            (14, 134),
            (14, 135),
            (14, 136),
            (14, 137),
            (14, 138),
            (14, 139),
            (14, 140),
            (14, 141),
            (14, 142),
            (14, 143),
            (14, 144),
            (14, 145),
            (14, 146),
            (14, 147),
            (14, 148),
            (14, 149),
            (15, 63),
            (15, 64),
            (15, 65),
            (15, 66),
            (15, 67),
            (15, 68),
            (15, 69),
            (15, 70),
            (15, 71),
            (15, 72),
            (15, 73),
            (15, 74),
            (15, 75),
            (15, 76),
            (15, 77),
            (15, 78),
            (15, 79),
            (15, 80),
            (15, 81),
            (15, 82),
            (15, 83),
            (15, 84),
            (15, 85),
            (15, 86),
            (15, 87),
            (15, 88),
            (15, 89),
            (15, 90),
            (15, 91),
            (15, 92),
            (15, 93),
            (15, 94),
            (15, 95),
            (15, 96),
            (15, 97),
            (15, 98),
            (15, 99),
            (15, 100),
            (15, 101),
            (15, 102),
            (15, 103),
            (15, 104),
            (15, 105),
            (15, 106),
            (15, 107),
            (15, 108),
            (15, 109),
            (15, 110),
            (15, 111),
            (15, 112),
            (15, 113),
            (15, 114),
            (15, 115),
            (15, 116),
            (15, 117),
            (15, 118),
            (15, 119),
            (15, 120),
            (15, 121),
            (15, 122),
            (15, 123),
            (15, 124),
            (15, 125),
            (15, 126),
            (15, 127),
            (15, 128),
            (15, 129),
            (15, 130),
            (15, 131),
            (15, 132),
            (15, 133),
            (15, 134),
            (15, 135),
            (15, 136),
            (15, 137),
            (15, 138),
            (15, 139),
            (15, 140),
            (15, 141),
            (15, 142),
            (15, 143),
            (15, 144),
            (15, 145),
            (15, 146),
            (15, 147),
            (15, 148),
            (15, 149),
            (16, 67),
            (16, 68),
            (16, 69),
            (16, 70),
            (16, 71),
            (16, 72),
            (16, 73),
            (16, 74),
            (16, 75),
            (16, 76),
            (16, 77),
            (16, 78),
            (16, 79),
            (16, 80),
            (16, 81),
            (16, 82),
            (16, 83),
            (16, 84),
            (16, 85),
            (16, 86),
            (16, 87),
            (16, 88),
            (16, 89),
            (16, 90),
            (16, 91),
            (16, 92),
            (16, 93),
            (16, 94),
            (16, 95),
            (16, 96),
            (16, 97),
            (16, 98),
            (16, 99),
            (16, 100),
            (16, 101),
            (16, 102),
            (16, 103),
            (16, 104),
            (16, 105),
            (16, 106),
            (16, 107),
            (16, 108),
            (16, 109),
            (16, 110),
            (16, 111),
            (16, 112),
            (16, 113),
            (16, 114),
            (16, 115),
            (16, 116),
            (16, 117),
            (16, 118),
            (16, 119),
            (16, 120),
            (16, 121),
            (16, 122),
            (16, 123),
            (16, 124),
            (16, 125),
            (16, 126),
            (16, 127),
            (16, 128),
            (16, 129),
            (16, 130),
            (16, 131),
            (16, 132),
            (16, 133),
            (16, 134),
            (16, 135),
            (16, 136),
            (16, 137),
            (16, 138),
            (16, 139),
            (16, 140),
            (16, 141),
            (16, 142),
            (16, 143),
            (16, 144),
            (16, 145),
            (16, 146),
            (16, 147),
            (16, 148),
            (16, 149),
            (17, 71),
            (17, 72),
            (17, 73),
            (17, 74),
            (17, 75),
            (17, 76),
            (17, 77),
            (17, 78),
            (17, 79),
            (17, 80),
            (17, 81),
            (17, 82),
            (17, 83),
            (17, 84),
            (17, 85),
            (17, 86),
            (17, 87),
            (17, 88),
            (17, 89),
            (17, 90),
            (17, 91),
            (17, 92),
            (17, 93),
            (17, 94),
            (17, 95),
            (17, 96),
            (17, 97),
            (17, 98),
            (17, 99),
            (17, 100),
            (17, 101),
            (17, 102),
            (17, 103),
            (17, 104),
            (17, 105),
            (17, 106),
            (17, 107),
            (17, 108),
            (17, 109),
            (17, 110),
            (17, 111),
            (17, 112),
            (17, 113),
            (17, 114),
            (17, 115),
            (17, 116),
            (17, 117),
            (17, 118),
            (17, 119),
            (17, 120),
            (17, 121),
            (17, 122),
            (17, 123),
            (17, 124),
            (17, 125),
            (17, 126),
            (17, 127),
            (17, 128),
            (17, 129),
            (17, 130),
            (17, 131),
            (17, 132),
            (17, 133),
            (17, 134),
            (17, 135),
            (17, 136),
            (17, 137),
            (17, 138),
            (17, 139),
            (17, 140),
            (17, 141),
            (17, 142),
            (17, 143),
            (17, 144),
            (17, 145),
            (17, 146),
            (17, 147),
            (17, 148),
            (17, 149),
            (18, 75),
            (18, 76),
            (18, 77),
            (18, 78),
            (18, 79),
            (18, 80),
            (18, 81),
            (18, 82),
            (18, 83),
            (18, 84),
            (18, 85),
            (18, 86),
            (18, 87),
            (18, 88),
            (18, 89),
            (18, 90),
            (18, 91),
            (18, 92),
            (18, 93),
            (18, 94),
            (18, 95),
            (18, 96),
            (18, 97),
            (18, 98),
            (18, 99),
            (18, 100),
            (18, 101),
            (18, 102),
            (18, 103),
            (18, 104),
            (18, 105),
            (18, 106),
            (18, 107),
            (18, 108),
            (18, 109),
            (18, 110),
            (18, 111),
            (18, 112),
            (18, 113),
            (18, 114),
            (18, 115),
            (18, 116),
            (18, 117),
            (18, 118),
            (18, 119),
            (18, 120),
            (18, 121),
            (18, 122),
            (18, 123),
            (18, 124),
            (18, 125),
            (18, 126),
            (18, 127),
            (18, 128),
            (18, 129),
            (18, 130),
            (18, 131),
            (18, 132),
            (18, 133),
            (18, 134),
            (18, 135),
            (18, 136),
            (18, 137),
            (18, 138),
            (18, 139),
            (18, 140),
            (18, 141),
            (18, 142),
            (18, 143),
            (18, 144),
            (18, 145),
            (18, 146),
            (18, 147),
            (18, 148),
            (18, 149),
            (19, 79),
            (19, 80),
            (19, 81),
            (19, 82),
            (19, 83),
            (19, 84),
            (19, 85),
            (19, 86),
            (19, 87),
            (19, 88),
            (19, 89),
            (19, 90),
            (19, 91),
            (19, 92),
            (19, 93),
            (19, 94),
            (19, 95),
            (19, 96),
            (19, 97),
            (19, 98),
            (19, 99),
            (19, 100),
            (19, 101),
            (19, 102),
            (19, 103),
            (19, 104),
            (19, 105),
            (19, 106),
            (19, 107),
            (19, 108),
            (19, 109),
            (19, 110),
            (19, 111),
            (19, 112),
            (19, 113),
            (19, 114),
            (19, 115),
            (19, 116),
            (19, 117),
            (19, 118),
            (19, 119),
            (19, 120),
            (19, 121),
            (19, 122),
            (19, 123),
            (19, 124),
            (19, 125),
            (19, 126),
            (19, 127),
            (19, 128),
            (19, 129),
            (19, 130),
            (19, 131),
            (19, 132),
            (19, 133),
            (19, 134),
            (19, 135),
            (19, 136),
            (19, 137),
            (19, 138),
            (19, 139),
            (19, 140),
            (19, 141),
            (19, 142),
            (19, 143),
            (19, 144),
            (19, 145),
            (19, 146),
            (19, 147),
            (19, 148),
            (19, 149),
            (20, 83),
            (20, 84),
            (20, 85),
            (20, 86),
            (20, 87),
            (20, 88),
            (20, 89),
            (20, 90),
            (20, 91),
            (20, 92),
            (20, 93),
            (20, 94),
            (20, 95),
            (20, 96),
            (20, 97),
            (20, 98),
            (20, 99),
            (20, 100),
            (20, 101),
            (20, 102),
            (20, 103),
            (20, 104),
            (20, 105),
            (20, 106),
            (20, 107),
            (20, 108),
            (20, 109),
            (20, 110),
            (20, 111),
            (20, 112),
            (20, 113),
            (20, 114),
            (20, 115),
            (20, 116),
            (20, 117),
            (20, 118),
            (20, 119),
            (20, 120),
            (20, 121),
            (20, 122),
            (20, 123),
            (20, 124),
            (20, 125),
            (20, 126),
            (20, 127),
            (20, 128),
            (20, 129),
            (20, 130),
            (20, 131),
            (20, 132),
            (20, 133),
            (20, 134),
            (20, 135),
            (20, 136),
            (20, 137),
            (20, 138),
            (20, 139),
            (20, 140),
            (20, 141),
            (20, 142),
            (20, 143),
            (20, 144),
            (20, 145),
            (20, 146),
            (20, 147),
            (20, 148),
            (20, 149),
            (21, 87),
            (21, 88),
            (21, 89),
            (21, 90),
            (21, 91),
            (21, 92),
            (21, 93),
            (21, 94),
            (21, 95),
            (21, 96),
            (21, 97),
            (21, 98),
            (21, 99),
            (21, 100),
            (21, 101),
            (21, 102),
            (21, 103),
            (21, 104),
            (21, 105),
            (21, 106),
            (21, 107),
            (21, 108),
            (21, 109),
            (21, 110),
            (21, 111),
            (21, 112),
            (21, 113),
            (21, 114),
            (21, 115),
            (21, 116),
            (21, 117),
            (21, 118),
            (21, 119),
            (21, 120),
            (21, 121),
            (21, 122),
            (21, 123),
            (21, 124),
            (21, 125),
            (21, 126),
            (21, 127),
            (21, 128),
            (21, 129),
            (21, 130),
            (21, 131),
            (21, 132),
            (21, 133),
            (21, 134),
            (21, 135),
            (21, 136),
            (21, 137),
            (21, 138),
            (21, 139),
            (21, 140),
            (21, 141),
            (21, 142),
            (21, 143),
            (21, 144),
            (21, 145),
            (21, 146),
            (21, 147),
            (21, 148),
            (21, 149),
            (22, 91),
            (22, 92),
            (22, 93),
            (22, 94),
            (22, 95),
            (22, 96),
            (22, 97),
            (22, 98),
            (22, 99),
            (22, 100),
            (22, 101),
            (22, 102),
            (22, 103),
            (22, 104),
            (22, 105),
            (22, 106),
            (22, 107),
            (22, 108),
            (22, 109),
            (22, 110),
            (22, 111),
            (22, 112),
            (22, 113),
            (22, 114),
            (22, 115),
            (22, 116),
            (22, 117),
            (22, 118),
            (22, 119),
            (22, 120),
            (22, 121),
            (22, 122),
            (22, 123),
            (22, 124),
            (22, 125),
            (22, 126),
            (22, 127),
            (22, 128),
            (22, 129),
            (22, 130),
            (22, 131),
            (22, 132),
            (22, 133),
            (22, 134),
            (22, 135),
            (22, 136),
            (22, 137),
            (22, 138),
            (22, 139),
            (22, 140),
            (22, 141),
            (22, 142),
            (22, 143),
            (22, 144),
            (22, 145),
            (22, 146),
            (22, 147),
            (22, 148),
            (22, 149),
            (23, 95),
            (23, 96),
            (23, 97),
            (23, 98),
            (23, 99),
            (23, 100),
            (23, 101),
            (23, 102),
            (23, 103),
            (23, 104),
            (23, 105),
            (23, 106),
            (23, 107),
            (23, 108),
            (23, 109),
            (23, 110),
            (23, 111),
            (23, 112),
            (23, 113),
            (23, 114),
            (23, 115),
            (23, 116),
            (23, 117),
            (23, 118),
            (23, 119),
            (23, 120),
            (23, 121),
            (23, 122),
            (23, 123),
            (23, 124),
            (23, 125),
            (23, 126),
            (23, 127),
            (23, 128),
            (23, 129),
            (23, 130),
            (23, 131),
            (23, 132),
            (23, 133),
            (23, 134),
            (23, 135),
            (23, 136),
            (23, 137),
            (23, 138),
            (23, 139),
            (23, 140),
            (23, 141),
            (23, 142),
            (23, 143),
            (23, 144),
            (23, 145),
            (23, 146),
            (23, 147),
            (23, 148),
            (23, 149),
            (24, 99),
            (24, 100),
            (24, 101),
            (24, 102),
            (24, 103),
            (24, 104),
            (24, 105),
            (24, 106),
            (24, 107),
            (24, 108),
            (24, 109),
            (24, 110),
            (24, 111),
            (24, 112),
            (24, 113),
            (24, 114),
            (24, 115),
            (24, 116),
            (24, 117),
            (24, 118),
            (24, 119),
            (24, 120),
            (24, 121),
            (24, 122),
            (24, 123),
            (24, 124),
            (24, 125),
            (24, 126),
            (24, 127),
            (24, 128),
            (24, 129),
            (24, 130),
            (24, 131),
            (24, 132),
            (24, 133),
            (24, 134),
            (24, 135),
            (24, 136),
            (24, 137),
            (24, 138),
            (24, 139),
            (24, 140),
            (24, 141),
            (24, 142),
            (24, 143),
            (24, 144),
            (24, 145),
            (24, 146),
            (24, 147),
            (24, 148),
            (24, 149),
            (25, 103),
            (25, 104),
            (25, 105),
            (25, 106),
            (25, 107),
            (25, 108),
            (25, 109),
            (25, 110),
            (25, 111),
            (25, 112),
            (25, 113),
            (25, 114),
            (25, 115),
            (25, 116),
            (25, 117),
            (25, 118),
            (25, 119),
            (25, 120),
            (25, 121),
            (25, 122),
            (25, 123),
            (25, 124),
            (25, 125),
            (25, 126),
            (25, 127),
            (25, 128),
            (25, 129),
            (25, 130),
            (25, 131),
            (25, 132),
            (25, 133),
            (25, 134),
            (25, 135),
            (25, 136),
            (25, 137),
            (25, 138),
            (25, 139),
            (25, 140),
            (25, 141),
            (25, 142),
            (25, 143),
            (25, 144),
            (25, 145),
            (25, 146),
            (25, 147),
            (25, 148),
            (25, 149),
            (26, 107),
            (26, 108),
            (26, 109),
            (26, 110),
            (26, 111),
            (26, 112),
            (26, 113),
            (26, 114),
            (26, 115),
            (26, 116),
            (26, 117),
            (26, 118),
            (26, 119),
            (26, 120),
            (26, 121),
            (26, 122),
            (26, 123),
            (26, 124),
            (26, 125),
            (26, 126),
            (26, 127),
            (26, 128),
            (26, 129),
            (26, 130),
            (26, 131),
            (26, 132),
            (26, 133),
            (26, 134),
            (26, 135),
            (26, 136),
            (26, 137),
            (26, 138),
            (26, 139),
            (26, 140),
            (26, 141),
            (26, 142),
            (26, 143),
            (26, 144),
            (26, 145),
            (26, 146),
            (26, 147),
            (26, 148),
            (26, 149),
            (27, 111),
            (27, 112),
            (27, 113),
            (27, 114),
            (27, 115),
            (27, 116),
            (27, 117),
            (27, 118),
            (27, 119),
            (27, 120),
            (27, 121),
            (27, 122),
            (27, 123),
            (27, 124),
            (27, 125),
            (27, 126),
            (27, 127),
            (27, 128),
            (27, 129),
            (27, 130),
            (27, 131),
            (27, 132),
            (27, 133),
            (27, 134),
            (27, 135),
            (27, 136),
            (27, 137),
            (27, 138),
            (27, 139),
            (27, 140),
            (27, 141),
            (27, 142),
            (27, 143),
            (27, 144),
            (27, 145),
            (27, 146),
            (27, 147),
            (27, 148),
            (27, 149),
            (28, 115),
            (28, 116),
            (28, 117),
            (28, 118),
            (28, 119),
            (28, 120),
            (28, 121),
            (28, 122),
            (28, 123),
            (28, 124),
            (28, 125),
            (28, 126),
            (28, 127),
            (28, 128),
            (28, 129),
            (28, 130),
            (28, 131),
            (28, 132),
            (28, 133),
            (28, 134),
            (28, 135),
            (28, 136),
            (28, 137),
            (28, 138),
            (28, 139),
            (28, 140),
            (28, 141),
            (28, 142),
            (28, 143),
            (28, 144),
            (28, 145),
            (28, 146),
            (28, 147),
            (28, 148),
            (28, 149),
            (29, 119),
            (29, 120),
            (29, 121),
            (29, 122),
            (29, 123),
            (29, 124),
            (29, 125),
            (29, 126),
            (29, 127),
            (29, 128),
            (29, 129),
            (29, 130),
            (29, 131),
            (29, 132),
            (29, 133),
            (29, 134),
            (29, 135),
            (29, 136),
            (29, 137),
            (29, 138),
            (29, 139),
            (29, 140),
            (29, 141),
            (29, 142),
            (29, 143),
            (29, 144),
            (29, 145),
            (29, 146),
            (29, 147),
            (29, 148),
            (29, 149),
            (30, 123),
            (30, 124),
            (30, 125),
            (30, 126),
            (30, 127),
            (30, 128),
            (30, 129),
            (30, 130),
            (30, 131),
            (30, 132),
            (30, 133),
            (30, 134),
            (30, 135),
            (30, 136),
            (30, 137),
            (30, 138),
            (30, 139),
            (30, 140),
            (30, 141),
            (30, 142),
            (30, 143),
            (30, 144),
            (30, 145),
            (30, 146),
            (30, 147),
            (30, 148),
            (30, 149),
            (31, 127),
            (31, 128),
            (31, 129),
            (31, 130),
            (31, 131),
            (31, 132),
            (31, 133),
            (31, 134),
            (31, 135),
            (31, 136),
            (31, 137),
            (31, 138),
            (31, 139),
            (31, 140),
            (31, 141),
            (31, 142),
            (31, 143),
            (31, 144),
            (31, 145),
            (31, 146),
            (31, 147),
            (31, 148),
            (31, 149),
            (32, 131),
            (32, 132),
            (32, 133),
            (32, 134),
            (32, 135),
            (32, 136),
            (32, 137),
            (32, 138),
            (32, 139),
            (32, 140),
            (32, 141),
            (32, 142),
            (32, 143),
            (32, 144),
            (32, 145),
            (32, 146),
            (32, 147),
            (32, 148),
            (32, 149),
            (33, 135),
            (33, 136),
            (33, 137),
            (33, 138),
            (33, 139),
            (33, 140),
            (33, 141),
            (33, 142),
            (33, 143),
            (33, 144),
            (33, 145),
            (33, 146),
            (33, 147),
            (33, 148),
            (33, 149),
            (34, 139),
            (34, 140),
            (34, 141),
            (34, 142),
            (34, 143),
            (34, 144),
            (34, 145),
            (34, 146),
            (34, 147),
            (34, 148),
            (34, 149),
            (35, 143),
            (35, 144),
            (35, 145),
            (35, 146),
            (35, 147),
            (35, 148),
            (35, 149),
            (36, 147),
            (36, 148),
            (36, 149),
        }:
            return 6
        return 22

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_1_output, num_attn_1_7_output):
        key = (num_attn_1_1_output, num_attn_1_7_output)
        return 43

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_7_outputs)
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


print(run(["<s>", "3"]))
