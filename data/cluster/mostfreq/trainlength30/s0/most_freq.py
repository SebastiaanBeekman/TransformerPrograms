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
        "output/length/rasp/mostfreq/trainlength30/s0/most_freq_weights.csv",
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
        if position in {0, 35, 38, 31}:
            return token == "2"
        elif position in {1}:
            return token == "0"
        elif position in {2, 3, 4, 5, 6}:
            return token == "1"
        elif position in {32, 33, 34, 36, 37, 7, 39, 10, 20, 21, 23, 30}:
            return token == ""
        elif position in {8}:
            return token == "3"
        elif position in {9, 14, 25}:
            return token == "5"
        elif position in {11, 15, 16, 19, 22, 26, 28}:
            return token == "4"
        elif position in {12, 13, 17, 18, 24, 27, 29}:
            return token == "<s>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 11, 14, 16, 25}:
            return token == "5"
        elif position in {1, 28, 29}:
            return token == "2"
        elif position in {2, 3, 4, 5, 6}:
            return token == "0"
        elif position in {7, 8, 10, 13, 17, 22, 23, 24, 26}:
            return token == "4"
        elif position in {9, 12, 15, 20, 21}:
            return token == "3"
        elif position in {32, 33, 34, 35, 36, 37, 38, 39, 18, 30, 31}:
            return token == ""
        elif position in {27, 19}:
            return token == "<s>"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {8, 3, 12, 15}:
            return k_position == 4
        elif q_position in {9, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {37, 6}:
            return k_position == 12
        elif q_position in {7}:
            return k_position == 19
        elif q_position in {10, 11, 13}:
            return k_position == 6
        elif q_position in {14}:
            return k_position == 11
        elif q_position in {16, 17, 18, 19, 20, 22, 23, 24, 26, 29}:
            return k_position == 0
        elif q_position in {21}:
            return k_position == 17
        elif q_position in {25}:
            return k_position == 10
        elif q_position in {27}:
            return k_position == 15
        elif q_position in {28}:
            return k_position == 9
        elif q_position in {30}:
            return k_position == 31
        elif q_position in {32, 34, 31}:
            return k_position == 25
        elif q_position in {33}:
            return k_position == 28
        elif q_position in {35}:
            return k_position == 37
        elif q_position in {36}:
            return k_position == 27
        elif q_position in {38}:
            return k_position == 36
        elif q_position in {39}:
            return k_position == 8

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 2, 8, 10, 13, 14, 15, 16, 19, 20, 21, 22, 24, 27, 28}:
            return token == "4"
        elif position in {1}:
            return token == "1"
        elif position in {3, 4, 5, 6, 29}:
            return token == "2"
        elif position in {26, 7}:
            return token == "3"
        elif position in {9, 11, 18, 23, 25}:
            return token == "5"
        elif position in {12}:
            return token == "<s>"
        elif position in {32, 33, 36, 37, 38, 39, 17, 30, 31}:
            return token == ""
        elif position in {34, 35}:
            return token == "<pad>"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 2, 3, 4, 10, 13, 17, 18}:
            return token == "4"
        elif position in {1}:
            return token == "1"
        elif position in {5, 8, 19, 26, 29}:
            return token == "2"
        elif position in {6}:
            return token == "0"
        elif position in {20, 37, 7}:
            return token == "3"
        elif position in {9, 11, 12, 14, 15, 16, 22}:
            return token == "5"
        elif position in {24, 27, 28, 21}:
            return token == "<s>"
        elif position in {32, 33, 34, 35, 36, 38, 39, 23, 25, 30, 31}:
            return token == ""

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3, 36}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 12
        elif q_position in {5}:
            return k_position == 14
        elif q_position in {6}:
            return k_position == 21
        elif q_position in {7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19}:
            return k_position == 3
        elif q_position in {16, 22, 25, 27, 28, 29}:
            return k_position == 24
        elif q_position in {18, 20, 21, 23, 24, 26}:
            return k_position == 19
        elif q_position in {35, 30}:
            return k_position == 39
        elif q_position in {31}:
            return k_position == 31
        elif q_position in {32}:
            return k_position == 30
        elif q_position in {33, 37, 39}:
            return k_position == 11
        elif q_position in {34}:
            return k_position == 38
        elif q_position in {38}:
            return k_position == 27

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, positions)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0, 9, 11, 14, 15, 18, 23}:
            return token == "5"
        elif position in {1, 2}:
            return token == "2"
        elif position in {3, 6}:
            return token == "0"
        elif position in {4, 5}:
            return token == "1"
        elif position in {7, 8, 10, 13, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29}:
            return token == "4"
        elif position in {12}:
            return token == "3"
        elif position in {32, 33, 34, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0}:
            return token == "4"
        elif position in {1}:
            return token == "0"
        elif position in {2, 3, 4, 5, 6}:
            return token == "3"
        elif position in {32, 33, 34, 35, 37, 38, 7, 39, 20, 30}:
            return token == ""
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
            21,
            22,
            23,
            24,
            25,
            26,
            28,
            29,
        }:
            return token == "1"
        elif position in {27}:
            return token == "2"
        elif position in {36, 31}:
            return token == "<pad>"

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 32, 2, 3, 4, 5, 6, 33, 34, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {1, 29, 25}:
            return token == "1"
        elif position in {28, 7}:
            return token == "2"
        elif position in {8, 9, 11, 12, 13, 14, 15, 19, 21, 23, 26}:
            return token == "4"
        elif position in {17, 10}:
            return token == "5"
        elif position in {16, 24, 27, 22}:
            return token == "3"
        elif position in {18, 20}:
            return token == "0"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 32, 2, 3, 4, 5, 6, 33, 34, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {1}:
            return token == "0"
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
        }:
            return token == "<s>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 1, 2, 3, 4}:
            return token == "2"
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

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 4, 5, 6}:
            return token == "<s>"
        elif position in {1, 2, 3}:
            return token == "1"
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
        elif position in {27}:
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
            13,
            17,
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
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {5, 6}:
            return token == "5"
        elif position in {7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 24}:
            return token == "<s>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 1}:
            return token == "4"
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
        elif position in {16, 19}:
            return token == "<pad>"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {
            0,
            2,
            3,
            4,
            5,
            12,
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
            38,
            39,
        }:
            return token == ""
        elif position in {1, 7}:
            return token == "3"
        elif position in {6, 8, 9, 10, 13}:
            return token == "<s>"
        elif position in {11, 37}:
            return token == "<pad>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {
            0,
            3,
            4,
            13,
            17,
            18,
            19,
            20,
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
        }:
            return token == ""
        elif position in {1}:
            return token == "5"
        elif position in {2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 21}:
            return token == "<s>"
        elif position in {36, 22}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, position):
        key = (attn_0_1_output, position)
        if key in {
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("0", 6),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("0", 15),
            ("0", 16),
            ("0", 17),
            ("0", 18),
            ("0", 19),
            ("0", 20),
            ("0", 21),
            ("0", 22),
            ("0", 23),
            ("0", 24),
            ("0", 25),
            ("0", 26),
            ("0", 27),
            ("0", 28),
            ("0", 29),
            ("0", 30),
            ("0", 31),
            ("0", 32),
            ("0", 33),
            ("0", 34),
            ("0", 35),
            ("0", 36),
            ("0", 37),
            ("0", 38),
            ("0", 39),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 30),
            ("1", 32),
            ("1", 35),
            ("1", 37),
            ("1", 38),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("2", 30),
            ("2", 31),
            ("2", 32),
            ("2", 33),
            ("2", 34),
            ("2", 35),
            ("2", 36),
            ("2", 37),
            ("2", 38),
            ("2", 39),
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("3", 30),
            ("3", 32),
            ("3", 33),
            ("3", 34),
            ("3", 35),
            ("3", 36),
            ("3", 37),
            ("3", 38),
            ("3", 39),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 30),
            ("4", 31),
            ("4", 32),
            ("4", 33),
            ("4", 34),
            ("4", 35),
            ("4", 36),
            ("4", 37),
            ("4", 38),
            ("4", 39),
            ("5", 4),
            ("5", 6),
            ("<s>", 6),
        }:
            return 1
        elif key in {
            ("0", 14),
            ("3", 0),
            ("3", 14),
            ("3", 31),
            ("<s>", 4),
            ("<s>", 31),
            ("<s>", 32),
            ("<s>", 33),
            ("<s>", 34),
            ("<s>", 36),
            ("<s>", 37),
            ("<s>", 38),
        }:
            return 8
        elif key in {
            ("0", 0),
            ("0", 7),
            ("0", 8),
            ("1", 0),
            ("1", 7),
            ("1", 8),
            ("1", 10),
            ("1", 17),
            ("1", 31),
            ("1", 33),
            ("1", 34),
            ("1", 36),
            ("1", 39),
            ("2", 0),
            ("2", 7),
            ("2", 8),
            ("5", 0),
            ("5", 7),
            ("5", 8),
            ("5", 10),
            ("5", 31),
            ("5", 33),
            ("5", 34),
            ("5", 36),
            ("5", 37),
            ("5", 38),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 10),
        }:
            return 0
        elif key in {
            ("0", 1),
            ("1", 1),
            ("2", 1),
            ("4", 0),
            ("4", 1),
            ("5", 1),
            ("<s>", 1),
        }:
            return 7
        elif key in {("<s>", 0), ("<s>", 14)}:
            return 14
        return 9

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_7_output):
        key = (attn_0_2_output, attn_0_7_output)
        if key in {("3", "3"), ("3", "4"), ("3", "5")}:
            return 10
        return 3

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_7_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_5_output, attn_0_2_output):
        key = (attn_0_5_output, attn_0_2_output)
        if key in {
            (0, "0"),
            (0, "1"),
            (2, "0"),
            (2, "1"),
            (3, "0"),
            (3, "1"),
            (6, "0"),
            (6, "1"),
            (7, "0"),
            (7, "1"),
            (8, "0"),
            (8, "1"),
            (9, "0"),
            (9, "1"),
            (10, "0"),
            (11, "0"),
            (11, "1"),
            (12, "0"),
            (12, "1"),
            (13, "0"),
            (13, "1"),
            (14, "0"),
            (14, "1"),
            (15, "0"),
            (15, "1"),
            (16, "0"),
            (16, "1"),
            (17, "0"),
            (17, "1"),
            (19, "0"),
            (19, "1"),
            (20, "0"),
            (20, "1"),
            (21, "0"),
            (21, "1"),
            (22, "0"),
            (22, "1"),
            (23, "0"),
            (23, "1"),
            (25, "0"),
            (25, "1"),
            (26, "0"),
            (26, "1"),
            (27, "0"),
            (27, "1"),
            (28, "0"),
            (28, "1"),
            (30, "0"),
            (30, "1"),
            (31, "0"),
            (31, "1"),
            (32, "0"),
            (32, "1"),
            (33, "0"),
            (33, "1"),
            (34, "0"),
            (34, "1"),
            (35, "0"),
            (35, "1"),
            (36, "0"),
            (36, "1"),
            (37, "0"),
            (37, "1"),
            (38, "0"),
            (38, "1"),
            (39, "0"),
            (39, "1"),
        }:
            return 0
        return 7

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_2_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_5_output, attn_0_6_output):
        key = (attn_0_5_output, attn_0_6_output)
        return 7

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_6_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output, num_attn_0_7_output):
        key = (num_attn_0_5_output, num_attn_0_7_output)
        return 7

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_4_output):
        key = (num_attn_0_1_output, num_attn_0_4_output)
        return 7

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 7

    num_mlp_0_2_outputs = [num_mlp_0_2(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 3

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 1
        elif q_position in {2, 8, 9, 10, 29}:
            return k_position == 3
        elif q_position in {
            3,
            6,
            11,
            12,
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
            28,
            38,
        }:
            return k_position == 0
        elif q_position in {4, 5}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {13}:
            return k_position == 22
        elif q_position in {27}:
            return k_position == 5
        elif q_position in {30}:
            return k_position == 38
        elif q_position in {31}:
            return k_position == 32
        elif q_position in {32, 34}:
            return k_position == 28
        elif q_position in {33}:
            return k_position == 26
        elif q_position in {35, 36}:
            return k_position == 24
        elif q_position in {37}:
            return k_position == 33
        elif q_position in {39}:
            return k_position == 37

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_2_output, token):
        if attn_0_2_output in {"5", "0", "2", "3", "1", "<s>"}:
            return token == "<s>"
        elif attn_0_2_output in {"4"}:
            return token == "0"

    attn_1_1_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_7_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_2_output, position):
        if attn_0_2_output in {"0", "2", "1", "<s>"}:
            return position == 0
        elif attn_0_2_output in {"3"}:
            return position == 10
        elif attn_0_2_output in {"4"}:
            return position == 2
        elif attn_0_2_output in {"5"}:
            return position == 3

    attn_1_2_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_2_output, token):
        if attn_0_2_output in {"0", "2", "<s>", "5"}:
            return token == ""
        elif attn_0_2_output in {"1"}:
            return token == "3"
        elif attn_0_2_output in {"3"}:
            return token == "2"
        elif attn_0_2_output in {"4"}:
            return token == "1"

    attn_1_3_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, token):
        if position in {0, 4, 6, 7}:
            return token == "<s>"
        elif position in {1}:
            return token == "2"
        elif position in {
            2,
            3,
            5,
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
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {34}:
            return token == "<pad>"

    attn_1_4_pattern = select_closest(tokens, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, mlp_0_0_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_token, k_token):
        if q_token in {"4", "0", "2"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"<s>", "3"}:
            return k_token == ""
        elif q_token in {"5"}:
            return k_token == "<s>"

    attn_1_5_pattern = select_closest(tokens, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_7_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"4", "1"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_1_6_pattern = select_closest(tokens, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_2_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, token):
        if position in {0}:
            return token == "4"
        elif position in {1, 2, 18, 23, 24, 25}:
            return token == "0"
        elif position in {3, 4, 37, 5, 6, 38}:
            return token == "<s>"
        elif position in {32, 33, 35, 36, 7, 8, 39, 10, 12, 13, 28, 30, 31}:
            return token == "5"
        elif position in {9, 14, 15, 16, 17, 21, 22, 26, 29}:
            return token == "1"
        elif position in {34, 11, 20}:
            return token == ""
        elif position in {19}:
            return token == "3"
        elif position in {27}:
            return token == "2"

    attn_1_7_pattern = select_closest(tokens, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, mlp_0_0_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(token, position):
        if token in {"0"}:
            return position == 22
        elif token in {"1"}:
            return position == 29
        elif token in {"2"}:
            return position == 4
        elif token in {"3"}:
            return position == 35
        elif token in {"4"}:
            return position == 36
        elif token in {"5"}:
            return position == 30
        elif token in {"<s>"}:
            return position == 11

    num_attn_1_0_pattern = select(positions, tokens, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, attn_0_3_output):
        if position in {0, 1, 2, 3, 4, 32, 33, 34, 35, 36, 37, 38, 39, 30, 31}:
            return attn_0_3_output == ""
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
            18,
            19,
            20,
            23,
            25,
            26,
            27,
            28,
            29,
        }:
            return attn_0_3_output == "4"
        elif position in {17, 21, 22}:
            return attn_0_3_output == "5"
        elif position in {24}:
            return attn_0_3_output == "3"

    num_attn_1_1_pattern = select(attn_0_3_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_5_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {
            0,
            2,
            3,
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
        elif position in {1}:
            return token == "3"
        elif position in {4}:
            return token == "<s>"
        elif position in {5, 6}:
            return token == "0"
        elif position in {27}:
            return token == "<pad>"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_3_output):
        if position in {0, 32, 2, 3, 4, 5, 6, 33, 34, 35, 36, 37, 38, 39, 22, 30, 31}:
            return attn_0_3_output == ""
        elif position in {1, 12}:
            return attn_0_3_output == "2"
        elif position in {7, 8, 9, 10, 14, 15, 21, 28, 29}:
            return attn_0_3_output == "4"
        elif position in {25, 11}:
            return attn_0_3_output == "1"
        elif position in {13, 23, 24, 26, 27}:
            return attn_0_3_output == "3"
        elif position in {16, 18, 19}:
            return attn_0_3_output == "0"
        elif position in {17}:
            return attn_0_3_output == "5"
        elif position in {20}:
            return attn_0_3_output == "<s>"

    num_attn_1_3_pattern = select(attn_0_3_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, attn_0_4_output):
        if position in {0, 1, 2, 32, 33, 5, 34, 35, 36, 37, 38, 39, 30, 31}:
            return attn_0_4_output == ""
        elif position in {3, 4}:
            return attn_0_4_output == "<pad>"
        elif position in {6}:
            return attn_0_4_output == "1"
        elif position in {
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
            18,
            20,
            21,
            22,
            23,
            24,
            28,
        }:
            return attn_0_4_output == "3"
        elif position in {27, 26, 11}:
            return attn_0_4_output == "4"
        elif position in {25, 19, 29}:
            return attn_0_4_output == "2"

    num_attn_1_4_pattern = select(attn_0_4_outputs, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_3_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_0_output, token):
        if attn_0_0_output in {"5", "0", "2", "4", "3", "<s>"}:
            return token == ""
        elif attn_0_0_output in {"1"}:
            return token == "1"

    num_attn_1_5_pattern = select(tokens, attn_0_0_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_3_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_2_output, attn_0_7_output):
        if attn_0_2_output in {"5", "0", "2", "3", "1"}:
            return attn_0_7_output == ""
        elif attn_0_2_output in {"4", "<s>"}:
            return attn_0_7_output == "4"

    num_attn_1_6_pattern = select(attn_0_7_outputs, attn_0_2_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_5_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, attn_0_1_output):
        if position in {0, 2, 36, 30}:
            return attn_0_1_output == "4"
        elif position in {1}:
            return attn_0_1_output == "1"
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
            23,
            24,
            25,
            26,
            28,
            29,
            32,
            37,
            38,
            39,
        }:
            return attn_0_1_output == ""
        elif position in {5, 6}:
            return attn_0_1_output == "2"
        elif position in {35, 27}:
            return attn_0_1_output == "<pad>"
        elif position in {33, 34, 31}:
            return attn_0_1_output == "5"

    num_attn_1_7_pattern = select(attn_0_1_outputs, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_3_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(num_mlp_0_0_output, num_mlp_0_2_output):
        key = (num_mlp_0_0_output, num_mlp_0_2_output)
        return 35

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, num_mlp_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_0_output, attn_1_5_output):
        key = (num_mlp_0_0_output, attn_1_5_output)
        return 25

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, attn_1_5_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(position, num_mlp_0_0_output):
        key = (position, num_mlp_0_0_output)
        if key in {(24, 22)}:
            return 27
        return 32

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(positions, num_mlp_0_0_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(num_mlp_0_3_output, attn_1_2_output):
        key = (num_mlp_0_3_output, attn_1_2_output)
        return 16

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(num_mlp_0_3_outputs, attn_1_2_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_6_output, num_attn_1_0_output):
        key = (num_attn_1_6_output, num_attn_1_0_output)
        return 1

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_4_output, num_attn_1_3_output):
        key = (num_attn_1_4_output, num_attn_1_3_output)
        return 13

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_3_output, num_attn_1_2_output):
        key = (num_attn_1_3_output, num_attn_1_2_output)
        return 17

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_7_output, num_attn_1_4_output):
        key = (num_attn_1_7_output, num_attn_1_4_output)
        return 10

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_1_output, attn_0_4_output):
        if attn_0_1_output in {"0", "3"}:
            return attn_0_4_output == "1"
        elif attn_0_1_output in {"1"}:
            return attn_0_4_output == "3"
        elif attn_0_1_output in {"2"}:
            return attn_0_4_output == "0"
        elif attn_0_1_output in {"4", "<s>", "5"}:
            return attn_0_4_output == ""

    attn_2_0_pattern = select_closest(attn_0_4_outputs, attn_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_4_output, position):
        if attn_0_4_output in {"0"}:
            return position == 8
        elif attn_0_4_output in {"2", "1", "<s>", "5"}:
            return position == 7
        elif attn_0_4_output in {"3"}:
            return position == 34
        elif attn_0_4_output in {"4"}:
            return position == 1

    attn_2_1_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "1"
        elif attn_0_2_output in {"1"}:
            return token == "0"
        elif attn_0_2_output in {"2"}:
            return token == "<s>"
        elif attn_0_2_output in {"3"}:
            return token == "4"
        elif attn_0_2_output in {"4"}:
            return token == "3"
        elif attn_0_2_output in {"<s>", "5"}:
            return token == ""

    attn_2_2_pattern = select_closest(tokens, attn_0_2_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, tokens)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "<s>"
        elif attn_0_4_output in {"1", "<s>"}:
            return token == "2"
        elif attn_0_4_output in {"2"}:
            return token == ""
        elif attn_0_4_output in {"3"}:
            return token == "1"
        elif attn_0_4_output in {"4"}:
            return token == "0"
        elif attn_0_4_output in {"5"}:
            return token == "5"

    attn_2_3_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_1_output, token):
        if attn_0_1_output in {"0", "5"}:
            return token == ""
        elif attn_0_1_output in {"1", "<s>"}:
            return token == "<s>"
        elif attn_0_1_output in {"2"}:
            return token == "3"
        elif attn_0_1_output in {"3"}:
            return token == "4"
        elif attn_0_1_output in {"4"}:
            return token == "2"

    attn_2_4_pattern = select_closest(tokens, attn_0_1_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_3_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_1_4_output, attn_0_0_output):
        if attn_1_4_output in {0, 33, 3, 4, 6}:
            return attn_0_0_output == "4"
        elif attn_1_4_output in {1, 7}:
            return attn_0_0_output == "<s>"
        elif attn_1_4_output in {2, 5, 8, 10, 12, 13, 17, 18, 20, 21, 23, 25, 27, 29}:
            return attn_0_0_output == ""
        elif attn_1_4_output in {34, 36, 38, 9, 11, 19, 22, 24, 28}:
            return attn_0_0_output == "1"
        elif attn_1_4_output in {14}:
            return attn_0_0_output == "0"
        elif attn_1_4_output in {39, 15}:
            return attn_0_0_output == "3"
        elif attn_1_4_output in {16, 35, 30}:
            return attn_0_0_output == "2"
        elif attn_1_4_output in {32, 26, 37, 31}:
            return attn_0_0_output == "5"

    attn_2_5_pattern = select_closest(attn_0_0_outputs, attn_1_4_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_0_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(position, mlp_0_0_output):
        if position in {0}:
            return mlp_0_0_output == 6
        elif position in {1, 2, 3}:
            return mlp_0_0_output == 7
        elif position in {32, 4, 5}:
            return mlp_0_0_output == 26
        elif position in {9, 6, 25}:
            return mlp_0_0_output == 10
        elif position in {7, 10, 11, 14, 17, 20, 22, 29}:
            return mlp_0_0_output == 19
        elif position in {8, 19, 12, 15}:
            return mlp_0_0_output == 18
        elif position in {13}:
            return mlp_0_0_output == 15
        elif position in {34, 16, 18, 21, 23, 27}:
            return mlp_0_0_output == 1
        elif position in {24, 26}:
            return mlp_0_0_output == 28
        elif position in {28, 39}:
            return mlp_0_0_output == 5
        elif position in {30}:
            return mlp_0_0_output == 0
        elif position in {31}:
            return mlp_0_0_output == 33
        elif position in {33}:
            return mlp_0_0_output == 32
        elif position in {35}:
            return mlp_0_0_output == 39
        elif position in {36}:
            return mlp_0_0_output == 34
        elif position in {37}:
            return mlp_0_0_output == 9
        elif position in {38}:
            return mlp_0_0_output == 17

    attn_2_6_pattern = select_closest(mlp_0_0_outputs, positions, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, mlp_0_1_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_3_output, token):
        if attn_0_3_output in {"5", "0", "2", "3", "1", "<s>"}:
            return token == "<s>"
        elif attn_0_3_output in {"4"}:
            return token == ""

    attn_2_7_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {"5", "0", "4", "3", "1"}:
            return attn_0_1_output == ""
        elif attn_0_3_output in {"2"}:
            return attn_0_1_output == "2"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_1_output == "<pad>"

    num_attn_2_0_pattern = select(attn_0_1_outputs, attn_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_6_output, attn_1_6_output):
        if attn_0_6_output in {"5", "0", "2", "4", "3", "<s>"}:
            return attn_1_6_output == ""
        elif attn_0_6_output in {"1"}:
            return attn_1_6_output == "1"

    num_attn_2_1_pattern = select(attn_1_6_outputs, attn_0_6_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, attn_0_4_output):
        if position in {
            0,
            1,
            2,
            3,
            4,
            11,
            12,
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
            37,
            39,
        }:
            return attn_0_4_output == ""
        elif position in {5, 6, 7}:
            return attn_0_4_output == "3"
        elif position in {36, 8, 9, 10, 13, 31}:
            return attn_0_4_output == "5"
        elif position in {38}:
            return attn_0_4_output == "1"

    num_attn_2_2_pattern = select(attn_0_4_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_6_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(token, attn_1_6_output):
        if token in {"5", "0", "2", "4", "1", "<s>"}:
            return attn_1_6_output == ""
        elif token in {"3"}:
            return attn_1_6_output == "3"

    num_attn_2_3_pattern = select(attn_1_6_outputs, tokens, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_6_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(token, position):
        if token in {"0"}:
            return position == 33
        elif token in {"1", "5"}:
            return position == 5
        elif token in {"2"}:
            return position == 38
        elif token in {"3"}:
            return position == 29
        elif token in {"4"}:
            return position == 35
        elif token in {"<s>"}:
            return position == 32

    num_attn_2_4_pattern = select(positions, tokens, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_7_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_6_output, attn_1_6_output):
        if attn_0_6_output in {"5", "0", "4", "1", "<s>"}:
            return attn_1_6_output == ""
        elif attn_0_6_output in {"2"}:
            return attn_1_6_output == "3"
        elif attn_0_6_output in {"3"}:
            return attn_1_6_output == "0"

    num_attn_2_5_pattern = select(attn_1_6_outputs, attn_0_6_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_5_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_4_output, attn_0_1_output):
        if attn_0_4_output in {"0"}:
            return attn_0_1_output == "2"
        elif attn_0_4_output in {"5", "2", "4", "3", "1", "<s>"}:
            return attn_0_1_output == ""

    num_attn_2_6_pattern = select(attn_0_1_outputs, attn_0_4_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_5_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_6_output, attn_1_0_output):
        if attn_0_6_output in {"5", "0", "4", "3", "1", "<s>"}:
            return attn_1_0_output == ""
        elif attn_0_6_output in {"2"}:
            return attn_1_0_output == "0"

    num_attn_2_7_pattern = select(attn_1_0_outputs, attn_0_6_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_1_2_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_3_output, mlp_0_0_output):
        key = (attn_1_3_output, mlp_0_0_output)
        return 27

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, mlp_0_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_0_2_output, num_mlp_1_1_output):
        key = (num_mlp_0_2_output, num_mlp_1_1_output)
        if key in {
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 14),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 32),
            (4, 33),
            (4, 36),
            (4, 39),
            (9, 16),
            (9, 17),
            (21, 6),
            (21, 11),
            (21, 17),
            (21, 27),
            (21, 33),
            (26, 1),
            (26, 6),
            (26, 11),
            (26, 16),
            (26, 17),
            (26, 26),
            (26, 27),
            (26, 33),
            (26, 36),
            (34, 11),
            (34, 17),
            (34, 27),
            (34, 33),
            (39, 11),
            (39, 16),
            (39, 17),
            (39, 27),
            (39, 33),
        }:
            return 7
        elif key in {
            (3, 39),
            (8, 39),
            (26, 39),
            (30, 39),
            (35, 39),
            (36, 39),
            (39, 39),
        }:
            return 3
        elif key in {(21, 16)}:
            return 34
        return 22

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_0_2_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_1_0_output, attn_1_1_output):
        key = (attn_1_0_output, attn_1_1_output)
        return 18

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_1_0_output, attn_0_3_output):
        key = (num_mlp_1_0_output, attn_0_3_output)
        return 4

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_1_0_outputs, attn_0_3_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_6_output, num_attn_2_5_output):
        key = (num_attn_0_6_output, num_attn_2_5_output)
        return 5

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_2_5_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_0_output, num_attn_0_2_output):
        key = (num_attn_2_0_output, num_attn_0_2_output)
        if key in {
            (3, 0),
            (4, 0),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 1),
            (6, 2),
            (7, 0),
            (7, 1),
            (7, 2),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 9),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (21, 9),
            (21, 10),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 10),
            (22, 11),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (23, 9),
            (23, 10),
            (23, 11),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 5),
            (24, 6),
            (24, 7),
            (24, 8),
            (24, 9),
            (24, 10),
            (24, 11),
            (24, 12),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (25, 5),
            (25, 6),
            (25, 7),
            (25, 8),
            (25, 9),
            (25, 10),
            (25, 11),
            (25, 12),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (26, 6),
            (26, 7),
            (26, 8),
            (26, 9),
            (26, 10),
            (26, 11),
            (26, 12),
            (26, 13),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (27, 7),
            (27, 8),
            (27, 9),
            (27, 10),
            (27, 11),
            (27, 12),
            (27, 13),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
            (28, 7),
            (28, 8),
            (28, 9),
            (28, 10),
            (28, 11),
            (28, 12),
            (28, 13),
            (28, 14),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (29, 7),
            (29, 8),
            (29, 9),
            (29, 10),
            (29, 11),
            (29, 12),
            (29, 13),
            (29, 14),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 8),
            (30, 9),
            (30, 10),
            (30, 11),
            (30, 12),
            (30, 13),
            (30, 14),
            (30, 15),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (31, 7),
            (31, 8),
            (31, 9),
            (31, 10),
            (31, 11),
            (31, 12),
            (31, 13),
            (31, 14),
            (31, 15),
            (31, 16),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (32, 7),
            (32, 8),
            (32, 9),
            (32, 10),
            (32, 11),
            (32, 12),
            (32, 13),
            (32, 14),
            (32, 15),
            (32, 16),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (33, 7),
            (33, 8),
            (33, 9),
            (33, 10),
            (33, 11),
            (33, 12),
            (33, 13),
            (33, 14),
            (33, 15),
            (33, 16),
            (33, 17),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 8),
            (34, 9),
            (34, 10),
            (34, 11),
            (34, 12),
            (34, 13),
            (34, 14),
            (34, 15),
            (34, 16),
            (34, 17),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (35, 8),
            (35, 9),
            (35, 10),
            (35, 11),
            (35, 12),
            (35, 13),
            (35, 14),
            (35, 15),
            (35, 16),
            (35, 17),
            (35, 18),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (36, 8),
            (36, 9),
            (36, 10),
            (36, 11),
            (36, 12),
            (36, 13),
            (36, 14),
            (36, 15),
            (36, 16),
            (36, 17),
            (36, 18),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (37, 8),
            (37, 9),
            (37, 10),
            (37, 11),
            (37, 12),
            (37, 13),
            (37, 14),
            (37, 15),
            (37, 16),
            (37, 17),
            (37, 18),
            (37, 19),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (38, 8),
            (38, 9),
            (38, 10),
            (38, 11),
            (38, 12),
            (38, 13),
            (38, 14),
            (38, 15),
            (38, 16),
            (38, 17),
            (38, 18),
            (38, 19),
            (38, 20),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (39, 9),
            (39, 10),
            (39, 11),
            (39, 12),
            (39, 13),
            (39, 14),
            (39, 15),
            (39, 16),
            (39, 17),
            (39, 18),
            (39, 19),
            (39, 20),
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
            (40, 17),
            (40, 18),
            (40, 19),
            (40, 20),
            (40, 21),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (41, 9),
            (41, 10),
            (41, 11),
            (41, 12),
            (41, 13),
            (41, 14),
            (41, 15),
            (41, 16),
            (41, 17),
            (41, 18),
            (41, 19),
            (41, 20),
            (41, 21),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (42, 9),
            (42, 10),
            (42, 11),
            (42, 12),
            (42, 13),
            (42, 14),
            (42, 15),
            (42, 16),
            (42, 17),
            (42, 18),
            (42, 19),
            (42, 20),
            (42, 21),
            (42, 22),
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
            (43, 10),
            (43, 11),
            (43, 12),
            (43, 13),
            (43, 14),
            (43, 15),
            (43, 16),
            (43, 17),
            (43, 18),
            (43, 19),
            (43, 20),
            (43, 21),
            (43, 22),
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
            (44, 10),
            (44, 11),
            (44, 12),
            (44, 13),
            (44, 14),
            (44, 15),
            (44, 16),
            (44, 17),
            (44, 18),
            (44, 19),
            (44, 20),
            (44, 21),
            (44, 22),
            (44, 23),
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
            (45, 10),
            (45, 11),
            (45, 12),
            (45, 13),
            (45, 14),
            (45, 15),
            (45, 16),
            (45, 17),
            (45, 18),
            (45, 19),
            (45, 20),
            (45, 21),
            (45, 22),
            (45, 23),
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
            (46, 11),
            (46, 12),
            (46, 13),
            (46, 14),
            (46, 15),
            (46, 16),
            (46, 17),
            (46, 18),
            (46, 19),
            (46, 20),
            (46, 21),
            (46, 22),
            (46, 23),
            (46, 24),
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
            (47, 11),
            (47, 12),
            (47, 13),
            (47, 14),
            (47, 15),
            (47, 16),
            (47, 17),
            (47, 18),
            (47, 19),
            (47, 20),
            (47, 21),
            (47, 22),
            (47, 23),
            (47, 24),
            (47, 25),
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
            (48, 11),
            (48, 12),
            (48, 13),
            (48, 14),
            (48, 15),
            (48, 16),
            (48, 17),
            (48, 18),
            (48, 19),
            (48, 20),
            (48, 21),
            (48, 22),
            (48, 23),
            (48, 24),
            (48, 25),
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
            (49, 11),
            (49, 12),
            (49, 13),
            (49, 14),
            (49, 15),
            (49, 16),
            (49, 17),
            (49, 18),
            (49, 19),
            (49, 20),
            (49, 21),
            (49, 22),
            (49, 23),
            (49, 24),
            (49, 25),
            (49, 26),
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
            (50, 12),
            (50, 13),
            (50, 14),
            (50, 15),
            (50, 16),
            (50, 17),
            (50, 18),
            (50, 19),
            (50, 20),
            (50, 21),
            (50, 22),
            (50, 23),
            (50, 24),
            (50, 25),
            (50, 26),
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
            (51, 12),
            (51, 13),
            (51, 14),
            (51, 15),
            (51, 16),
            (51, 17),
            (51, 18),
            (51, 19),
            (51, 20),
            (51, 21),
            (51, 22),
            (51, 23),
            (51, 24),
            (51, 25),
            (51, 26),
            (51, 27),
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
            (52, 12),
            (52, 13),
            (52, 14),
            (52, 15),
            (52, 16),
            (52, 17),
            (52, 18),
            (52, 19),
            (52, 20),
            (52, 21),
            (52, 22),
            (52, 23),
            (52, 24),
            (52, 25),
            (52, 26),
            (52, 27),
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
            (53, 13),
            (53, 14),
            (53, 15),
            (53, 16),
            (53, 17),
            (53, 18),
            (53, 19),
            (53, 20),
            (53, 21),
            (53, 22),
            (53, 23),
            (53, 24),
            (53, 25),
            (53, 26),
            (53, 27),
            (53, 28),
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
            (54, 13),
            (54, 14),
            (54, 15),
            (54, 16),
            (54, 17),
            (54, 18),
            (54, 19),
            (54, 20),
            (54, 21),
            (54, 22),
            (54, 23),
            (54, 24),
            (54, 25),
            (54, 26),
            (54, 27),
            (54, 28),
            (54, 29),
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
            (55, 13),
            (55, 14),
            (55, 15),
            (55, 16),
            (55, 17),
            (55, 18),
            (55, 19),
            (55, 20),
            (55, 21),
            (55, 22),
            (55, 23),
            (55, 24),
            (55, 25),
            (55, 26),
            (55, 27),
            (55, 28),
            (55, 29),
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
            (56, 13),
            (56, 14),
            (56, 15),
            (56, 16),
            (56, 17),
            (56, 18),
            (56, 19),
            (56, 20),
            (56, 21),
            (56, 22),
            (56, 23),
            (56, 24),
            (56, 25),
            (56, 26),
            (56, 27),
            (56, 28),
            (56, 29),
            (56, 30),
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
            (57, 14),
            (57, 15),
            (57, 16),
            (57, 17),
            (57, 18),
            (57, 19),
            (57, 20),
            (57, 21),
            (57, 22),
            (57, 23),
            (57, 24),
            (57, 25),
            (57, 26),
            (57, 27),
            (57, 28),
            (57, 29),
            (57, 30),
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
            (58, 14),
            (58, 15),
            (58, 16),
            (58, 17),
            (58, 18),
            (58, 19),
            (58, 20),
            (58, 21),
            (58, 22),
            (58, 23),
            (58, 24),
            (58, 25),
            (58, 26),
            (58, 27),
            (58, 28),
            (58, 29),
            (58, 30),
            (58, 31),
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
            (59, 14),
            (59, 15),
            (59, 16),
            (59, 17),
            (59, 18),
            (59, 19),
            (59, 20),
            (59, 21),
            (59, 22),
            (59, 23),
            (59, 24),
            (59, 25),
            (59, 26),
            (59, 27),
            (59, 28),
            (59, 29),
            (59, 30),
            (59, 31),
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
            (60, 15),
            (60, 16),
            (60, 17),
            (60, 18),
            (60, 19),
            (60, 20),
            (60, 21),
            (60, 22),
            (60, 23),
            (60, 24),
            (60, 25),
            (60, 26),
            (60, 27),
            (60, 28),
            (60, 29),
            (60, 30),
            (60, 31),
            (60, 32),
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
            (61, 15),
            (61, 16),
            (61, 17),
            (61, 18),
            (61, 19),
            (61, 20),
            (61, 21),
            (61, 22),
            (61, 23),
            (61, 24),
            (61, 25),
            (61, 26),
            (61, 27),
            (61, 28),
            (61, 29),
            (61, 30),
            (61, 31),
            (61, 32),
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
            (62, 15),
            (62, 16),
            (62, 17),
            (62, 18),
            (62, 19),
            (62, 20),
            (62, 21),
            (62, 22),
            (62, 23),
            (62, 24),
            (62, 25),
            (62, 26),
            (62, 27),
            (62, 28),
            (62, 29),
            (62, 30),
            (62, 31),
            (62, 32),
            (62, 33),
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
            (63, 15),
            (63, 16),
            (63, 17),
            (63, 18),
            (63, 19),
            (63, 20),
            (63, 21),
            (63, 22),
            (63, 23),
            (63, 24),
            (63, 25),
            (63, 26),
            (63, 27),
            (63, 28),
            (63, 29),
            (63, 30),
            (63, 31),
            (63, 32),
            (63, 33),
            (63, 34),
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
            (64, 16),
            (64, 17),
            (64, 18),
            (64, 19),
            (64, 20),
            (64, 21),
            (64, 22),
            (64, 23),
            (64, 24),
            (64, 25),
            (64, 26),
            (64, 27),
            (64, 28),
            (64, 29),
            (64, 30),
            (64, 31),
            (64, 32),
            (64, 33),
            (64, 34),
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
            (65, 16),
            (65, 17),
            (65, 18),
            (65, 19),
            (65, 20),
            (65, 21),
            (65, 22),
            (65, 23),
            (65, 24),
            (65, 25),
            (65, 26),
            (65, 27),
            (65, 28),
            (65, 29),
            (65, 30),
            (65, 31),
            (65, 32),
            (65, 33),
            (65, 34),
            (65, 35),
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
            (66, 16),
            (66, 17),
            (66, 18),
            (66, 19),
            (66, 20),
            (66, 21),
            (66, 22),
            (66, 23),
            (66, 24),
            (66, 25),
            (66, 26),
            (66, 27),
            (66, 28),
            (66, 29),
            (66, 30),
            (66, 31),
            (66, 32),
            (66, 33),
            (66, 34),
            (66, 35),
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
            (67, 17),
            (67, 18),
            (67, 19),
            (67, 20),
            (67, 21),
            (67, 22),
            (67, 23),
            (67, 24),
            (67, 25),
            (67, 26),
            (67, 27),
            (67, 28),
            (67, 29),
            (67, 30),
            (67, 31),
            (67, 32),
            (67, 33),
            (67, 34),
            (67, 35),
            (67, 36),
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
            (68, 17),
            (68, 18),
            (68, 19),
            (68, 20),
            (68, 21),
            (68, 22),
            (68, 23),
            (68, 24),
            (68, 25),
            (68, 26),
            (68, 27),
            (68, 28),
            (68, 29),
            (68, 30),
            (68, 31),
            (68, 32),
            (68, 33),
            (68, 34),
            (68, 35),
            (68, 36),
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
            (69, 17),
            (69, 18),
            (69, 19),
            (69, 20),
            (69, 21),
            (69, 22),
            (69, 23),
            (69, 24),
            (69, 25),
            (69, 26),
            (69, 27),
            (69, 28),
            (69, 29),
            (69, 30),
            (69, 31),
            (69, 32),
            (69, 33),
            (69, 34),
            (69, 35),
            (69, 36),
            (69, 37),
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
            (70, 17),
            (70, 18),
            (70, 19),
            (70, 20),
            (70, 21),
            (70, 22),
            (70, 23),
            (70, 24),
            (70, 25),
            (70, 26),
            (70, 27),
            (70, 28),
            (70, 29),
            (70, 30),
            (70, 31),
            (70, 32),
            (70, 33),
            (70, 34),
            (70, 35),
            (70, 36),
            (70, 37),
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
            (71, 18),
            (71, 19),
            (71, 20),
            (71, 21),
            (71, 22),
            (71, 23),
            (71, 24),
            (71, 25),
            (71, 26),
            (71, 27),
            (71, 28),
            (71, 29),
            (71, 30),
            (71, 31),
            (71, 32),
            (71, 33),
            (71, 34),
            (71, 35),
            (71, 36),
            (71, 37),
            (71, 38),
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
            (72, 18),
            (72, 19),
            (72, 20),
            (72, 21),
            (72, 22),
            (72, 23),
            (72, 24),
            (72, 25),
            (72, 26),
            (72, 27),
            (72, 28),
            (72, 29),
            (72, 30),
            (72, 31),
            (72, 32),
            (72, 33),
            (72, 34),
            (72, 35),
            (72, 36),
            (72, 37),
            (72, 38),
            (72, 39),
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
            (73, 18),
            (73, 19),
            (73, 20),
            (73, 21),
            (73, 22),
            (73, 23),
            (73, 24),
            (73, 25),
            (73, 26),
            (73, 27),
            (73, 28),
            (73, 29),
            (73, 30),
            (73, 31),
            (73, 32),
            (73, 33),
            (73, 34),
            (73, 35),
            (73, 36),
            (73, 37),
            (73, 38),
            (73, 39),
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
            (74, 19),
            (74, 20),
            (74, 21),
            (74, 22),
            (74, 23),
            (74, 24),
            (74, 25),
            (74, 26),
            (74, 27),
            (74, 28),
            (74, 29),
            (74, 30),
            (74, 31),
            (74, 32),
            (74, 33),
            (74, 34),
            (74, 35),
            (74, 36),
            (74, 37),
            (74, 38),
            (74, 39),
            (74, 40),
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
            (75, 19),
            (75, 20),
            (75, 21),
            (75, 22),
            (75, 23),
            (75, 24),
            (75, 25),
            (75, 26),
            (75, 27),
            (75, 28),
            (75, 29),
            (75, 30),
            (75, 31),
            (75, 32),
            (75, 33),
            (75, 34),
            (75, 35),
            (75, 36),
            (75, 37),
            (75, 38),
            (75, 39),
            (75, 40),
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
            (76, 19),
            (76, 20),
            (76, 21),
            (76, 22),
            (76, 23),
            (76, 24),
            (76, 25),
            (76, 26),
            (76, 27),
            (76, 28),
            (76, 29),
            (76, 30),
            (76, 31),
            (76, 32),
            (76, 33),
            (76, 34),
            (76, 35),
            (76, 36),
            (76, 37),
            (76, 38),
            (76, 39),
            (76, 40),
            (76, 41),
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
            (77, 19),
            (77, 20),
            (77, 21),
            (77, 22),
            (77, 23),
            (77, 24),
            (77, 25),
            (77, 26),
            (77, 27),
            (77, 28),
            (77, 29),
            (77, 30),
            (77, 31),
            (77, 32),
            (77, 33),
            (77, 34),
            (77, 35),
            (77, 36),
            (77, 37),
            (77, 38),
            (77, 39),
            (77, 40),
            (77, 41),
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
            (78, 20),
            (78, 21),
            (78, 22),
            (78, 23),
            (78, 24),
            (78, 25),
            (78, 26),
            (78, 27),
            (78, 28),
            (78, 29),
            (78, 30),
            (78, 31),
            (78, 32),
            (78, 33),
            (78, 34),
            (78, 35),
            (78, 36),
            (78, 37),
            (78, 38),
            (78, 39),
            (78, 40),
            (78, 41),
            (78, 42),
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
            (79, 20),
            (79, 21),
            (79, 22),
            (79, 23),
            (79, 24),
            (79, 25),
            (79, 26),
            (79, 27),
            (79, 28),
            (79, 29),
            (79, 30),
            (79, 31),
            (79, 32),
            (79, 33),
            (79, 34),
            (79, 35),
            (79, 36),
            (79, 37),
            (79, 38),
            (79, 39),
            (79, 40),
            (79, 41),
            (79, 42),
            (79, 43),
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
            (80, 20),
            (80, 21),
            (80, 22),
            (80, 23),
            (80, 24),
            (80, 25),
            (80, 26),
            (80, 27),
            (80, 28),
            (80, 29),
            (80, 30),
            (80, 31),
            (80, 32),
            (80, 33),
            (80, 34),
            (80, 35),
            (80, 36),
            (80, 37),
            (80, 38),
            (80, 39),
            (80, 40),
            (80, 41),
            (80, 42),
            (80, 43),
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
            (81, 21),
            (81, 22),
            (81, 23),
            (81, 24),
            (81, 25),
            (81, 26),
            (81, 27),
            (81, 28),
            (81, 29),
            (81, 30),
            (81, 31),
            (81, 32),
            (81, 33),
            (81, 34),
            (81, 35),
            (81, 36),
            (81, 37),
            (81, 38),
            (81, 39),
            (81, 40),
            (81, 41),
            (81, 42),
            (81, 43),
            (81, 44),
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
            (82, 21),
            (82, 22),
            (82, 23),
            (82, 24),
            (82, 25),
            (82, 26),
            (82, 27),
            (82, 28),
            (82, 29),
            (82, 30),
            (82, 31),
            (82, 32),
            (82, 33),
            (82, 34),
            (82, 35),
            (82, 36),
            (82, 37),
            (82, 38),
            (82, 39),
            (82, 40),
            (82, 41),
            (82, 42),
            (82, 43),
            (82, 44),
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
            (83, 21),
            (83, 22),
            (83, 23),
            (83, 24),
            (83, 25),
            (83, 26),
            (83, 27),
            (83, 28),
            (83, 29),
            (83, 30),
            (83, 31),
            (83, 32),
            (83, 33),
            (83, 34),
            (83, 35),
            (83, 36),
            (83, 37),
            (83, 38),
            (83, 39),
            (83, 40),
            (83, 41),
            (83, 42),
            (83, 43),
            (83, 44),
            (83, 45),
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
            (84, 21),
            (84, 22),
            (84, 23),
            (84, 24),
            (84, 25),
            (84, 26),
            (84, 27),
            (84, 28),
            (84, 29),
            (84, 30),
            (84, 31),
            (84, 32),
            (84, 33),
            (84, 34),
            (84, 35),
            (84, 36),
            (84, 37),
            (84, 38),
            (84, 39),
            (84, 40),
            (84, 41),
            (84, 42),
            (84, 43),
            (84, 44),
            (84, 45),
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
            (85, 22),
            (85, 23),
            (85, 24),
            (85, 25),
            (85, 26),
            (85, 27),
            (85, 28),
            (85, 29),
            (85, 30),
            (85, 31),
            (85, 32),
            (85, 33),
            (85, 34),
            (85, 35),
            (85, 36),
            (85, 37),
            (85, 38),
            (85, 39),
            (85, 40),
            (85, 41),
            (85, 42),
            (85, 43),
            (85, 44),
            (85, 45),
            (85, 46),
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
            (86, 22),
            (86, 23),
            (86, 24),
            (86, 25),
            (86, 26),
            (86, 27),
            (86, 28),
            (86, 29),
            (86, 30),
            (86, 31),
            (86, 32),
            (86, 33),
            (86, 34),
            (86, 35),
            (86, 36),
            (86, 37),
            (86, 38),
            (86, 39),
            (86, 40),
            (86, 41),
            (86, 42),
            (86, 43),
            (86, 44),
            (86, 45),
            (86, 46),
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
            (87, 22),
            (87, 23),
            (87, 24),
            (87, 25),
            (87, 26),
            (87, 27),
            (87, 28),
            (87, 29),
            (87, 30),
            (87, 31),
            (87, 32),
            (87, 33),
            (87, 34),
            (87, 35),
            (87, 36),
            (87, 37),
            (87, 38),
            (87, 39),
            (87, 40),
            (87, 41),
            (87, 42),
            (87, 43),
            (87, 44),
            (87, 45),
            (87, 46),
            (87, 47),
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
            (88, 23),
            (88, 24),
            (88, 25),
            (88, 26),
            (88, 27),
            (88, 28),
            (88, 29),
            (88, 30),
            (88, 31),
            (88, 32),
            (88, 33),
            (88, 34),
            (88, 35),
            (88, 36),
            (88, 37),
            (88, 38),
            (88, 39),
            (88, 40),
            (88, 41),
            (88, 42),
            (88, 43),
            (88, 44),
            (88, 45),
            (88, 46),
            (88, 47),
            (88, 48),
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
            (89, 23),
            (89, 24),
            (89, 25),
            (89, 26),
            (89, 27),
            (89, 28),
            (89, 29),
            (89, 30),
            (89, 31),
            (89, 32),
            (89, 33),
            (89, 34),
            (89, 35),
            (89, 36),
            (89, 37),
            (89, 38),
            (89, 39),
            (89, 40),
            (89, 41),
            (89, 42),
            (89, 43),
            (89, 44),
            (89, 45),
            (89, 46),
            (89, 47),
            (89, 48),
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
            (90, 23),
            (90, 24),
            (90, 25),
            (90, 26),
            (90, 27),
            (90, 28),
            (90, 29),
            (90, 30),
            (90, 31),
            (90, 32),
            (90, 33),
            (90, 34),
            (90, 35),
            (90, 36),
            (90, 37),
            (90, 38),
            (90, 39),
            (90, 40),
            (90, 41),
            (90, 42),
            (90, 43),
            (90, 44),
            (90, 45),
            (90, 46),
            (90, 47),
            (90, 48),
            (90, 49),
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
            (91, 23),
            (91, 24),
            (91, 25),
            (91, 26),
            (91, 27),
            (91, 28),
            (91, 29),
            (91, 30),
            (91, 31),
            (91, 32),
            (91, 33),
            (91, 34),
            (91, 35),
            (91, 36),
            (91, 37),
            (91, 38),
            (91, 39),
            (91, 40),
            (91, 41),
            (91, 42),
            (91, 43),
            (91, 44),
            (91, 45),
            (91, 46),
            (91, 47),
            (91, 48),
            (91, 49),
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
            (92, 24),
            (92, 25),
            (92, 26),
            (92, 27),
            (92, 28),
            (92, 29),
            (92, 30),
            (92, 31),
            (92, 32),
            (92, 33),
            (92, 34),
            (92, 35),
            (92, 36),
            (92, 37),
            (92, 38),
            (92, 39),
            (92, 40),
            (92, 41),
            (92, 42),
            (92, 43),
            (92, 44),
            (92, 45),
            (92, 46),
            (92, 47),
            (92, 48),
            (92, 49),
            (92, 50),
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
            (93, 24),
            (93, 25),
            (93, 26),
            (93, 27),
            (93, 28),
            (93, 29),
            (93, 30),
            (93, 31),
            (93, 32),
            (93, 33),
            (93, 34),
            (93, 35),
            (93, 36),
            (93, 37),
            (93, 38),
            (93, 39),
            (93, 40),
            (93, 41),
            (93, 42),
            (93, 43),
            (93, 44),
            (93, 45),
            (93, 46),
            (93, 47),
            (93, 48),
            (93, 49),
            (93, 50),
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
            (94, 24),
            (94, 25),
            (94, 26),
            (94, 27),
            (94, 28),
            (94, 29),
            (94, 30),
            (94, 31),
            (94, 32),
            (94, 33),
            (94, 34),
            (94, 35),
            (94, 36),
            (94, 37),
            (94, 38),
            (94, 39),
            (94, 40),
            (94, 41),
            (94, 42),
            (94, 43),
            (94, 44),
            (94, 45),
            (94, 46),
            (94, 47),
            (94, 48),
            (94, 49),
            (94, 50),
            (94, 51),
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
            (95, 25),
            (95, 26),
            (95, 27),
            (95, 28),
            (95, 29),
            (95, 30),
            (95, 31),
            (95, 32),
            (95, 33),
            (95, 34),
            (95, 35),
            (95, 36),
            (95, 37),
            (95, 38),
            (95, 39),
            (95, 40),
            (95, 41),
            (95, 42),
            (95, 43),
            (95, 44),
            (95, 45),
            (95, 46),
            (95, 47),
            (95, 48),
            (95, 49),
            (95, 50),
            (95, 51),
            (95, 52),
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
            (96, 25),
            (96, 26),
            (96, 27),
            (96, 28),
            (96, 29),
            (96, 30),
            (96, 31),
            (96, 32),
            (96, 33),
            (96, 34),
            (96, 35),
            (96, 36),
            (96, 37),
            (96, 38),
            (96, 39),
            (96, 40),
            (96, 41),
            (96, 42),
            (96, 43),
            (96, 44),
            (96, 45),
            (96, 46),
            (96, 47),
            (96, 48),
            (96, 49),
            (96, 50),
            (96, 51),
            (96, 52),
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
            (97, 25),
            (97, 26),
            (97, 27),
            (97, 28),
            (97, 29),
            (97, 30),
            (97, 31),
            (97, 32),
            (97, 33),
            (97, 34),
            (97, 35),
            (97, 36),
            (97, 37),
            (97, 38),
            (97, 39),
            (97, 40),
            (97, 41),
            (97, 42),
            (97, 43),
            (97, 44),
            (97, 45),
            (97, 46),
            (97, 47),
            (97, 48),
            (97, 49),
            (97, 50),
            (97, 51),
            (97, 52),
            (97, 53),
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
            (98, 25),
            (98, 26),
            (98, 27),
            (98, 28),
            (98, 29),
            (98, 30),
            (98, 31),
            (98, 32),
            (98, 33),
            (98, 34),
            (98, 35),
            (98, 36),
            (98, 37),
            (98, 38),
            (98, 39),
            (98, 40),
            (98, 41),
            (98, 42),
            (98, 43),
            (98, 44),
            (98, 45),
            (98, 46),
            (98, 47),
            (98, 48),
            (98, 49),
            (98, 50),
            (98, 51),
            (98, 52),
            (98, 53),
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
            (99, 26),
            (99, 27),
            (99, 28),
            (99, 29),
            (99, 30),
            (99, 31),
            (99, 32),
            (99, 33),
            (99, 34),
            (99, 35),
            (99, 36),
            (99, 37),
            (99, 38),
            (99, 39),
            (99, 40),
            (99, 41),
            (99, 42),
            (99, 43),
            (99, 44),
            (99, 45),
            (99, 46),
            (99, 47),
            (99, 48),
            (99, 49),
            (99, 50),
            (99, 51),
            (99, 52),
            (99, 53),
            (99, 54),
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
            (100, 26),
            (100, 27),
            (100, 28),
            (100, 29),
            (100, 30),
            (100, 31),
            (100, 32),
            (100, 33),
            (100, 34),
            (100, 35),
            (100, 36),
            (100, 37),
            (100, 38),
            (100, 39),
            (100, 40),
            (100, 41),
            (100, 42),
            (100, 43),
            (100, 44),
            (100, 45),
            (100, 46),
            (100, 47),
            (100, 48),
            (100, 49),
            (100, 50),
            (100, 51),
            (100, 52),
            (100, 53),
            (100, 54),
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
            (101, 26),
            (101, 27),
            (101, 28),
            (101, 29),
            (101, 30),
            (101, 31),
            (101, 32),
            (101, 33),
            (101, 34),
            (101, 35),
            (101, 36),
            (101, 37),
            (101, 38),
            (101, 39),
            (101, 40),
            (101, 41),
            (101, 42),
            (101, 43),
            (101, 44),
            (101, 45),
            (101, 46),
            (101, 47),
            (101, 48),
            (101, 49),
            (101, 50),
            (101, 51),
            (101, 52),
            (101, 53),
            (101, 54),
            (101, 55),
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
            (102, 27),
            (102, 28),
            (102, 29),
            (102, 30),
            (102, 31),
            (102, 32),
            (102, 33),
            (102, 34),
            (102, 35),
            (102, 36),
            (102, 37),
            (102, 38),
            (102, 39),
            (102, 40),
            (102, 41),
            (102, 42),
            (102, 43),
            (102, 44),
            (102, 45),
            (102, 46),
            (102, 47),
            (102, 48),
            (102, 49),
            (102, 50),
            (102, 51),
            (102, 52),
            (102, 53),
            (102, 54),
            (102, 55),
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
            (103, 27),
            (103, 28),
            (103, 29),
            (103, 30),
            (103, 31),
            (103, 32),
            (103, 33),
            (103, 34),
            (103, 35),
            (103, 36),
            (103, 37),
            (103, 38),
            (103, 39),
            (103, 40),
            (103, 41),
            (103, 42),
            (103, 43),
            (103, 44),
            (103, 45),
            (103, 46),
            (103, 47),
            (103, 48),
            (103, 49),
            (103, 50),
            (103, 51),
            (103, 52),
            (103, 53),
            (103, 54),
            (103, 55),
            (103, 56),
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
            (104, 27),
            (104, 28),
            (104, 29),
            (104, 30),
            (104, 31),
            (104, 32),
            (104, 33),
            (104, 34),
            (104, 35),
            (104, 36),
            (104, 37),
            (104, 38),
            (104, 39),
            (104, 40),
            (104, 41),
            (104, 42),
            (104, 43),
            (104, 44),
            (104, 45),
            (104, 46),
            (104, 47),
            (104, 48),
            (104, 49),
            (104, 50),
            (104, 51),
            (104, 52),
            (104, 53),
            (104, 54),
            (104, 55),
            (104, 56),
            (104, 57),
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
            (105, 27),
            (105, 28),
            (105, 29),
            (105, 30),
            (105, 31),
            (105, 32),
            (105, 33),
            (105, 34),
            (105, 35),
            (105, 36),
            (105, 37),
            (105, 38),
            (105, 39),
            (105, 40),
            (105, 41),
            (105, 42),
            (105, 43),
            (105, 44),
            (105, 45),
            (105, 46),
            (105, 47),
            (105, 48),
            (105, 49),
            (105, 50),
            (105, 51),
            (105, 52),
            (105, 53),
            (105, 54),
            (105, 55),
            (105, 56),
            (105, 57),
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
            (106, 28),
            (106, 29),
            (106, 30),
            (106, 31),
            (106, 32),
            (106, 33),
            (106, 34),
            (106, 35),
            (106, 36),
            (106, 37),
            (106, 38),
            (106, 39),
            (106, 40),
            (106, 41),
            (106, 42),
            (106, 43),
            (106, 44),
            (106, 45),
            (106, 46),
            (106, 47),
            (106, 48),
            (106, 49),
            (106, 50),
            (106, 51),
            (106, 52),
            (106, 53),
            (106, 54),
            (106, 55),
            (106, 56),
            (106, 57),
            (106, 58),
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
            (107, 28),
            (107, 29),
            (107, 30),
            (107, 31),
            (107, 32),
            (107, 33),
            (107, 34),
            (107, 35),
            (107, 36),
            (107, 37),
            (107, 38),
            (107, 39),
            (107, 40),
            (107, 41),
            (107, 42),
            (107, 43),
            (107, 44),
            (107, 45),
            (107, 46),
            (107, 47),
            (107, 48),
            (107, 49),
            (107, 50),
            (107, 51),
            (107, 52),
            (107, 53),
            (107, 54),
            (107, 55),
            (107, 56),
            (107, 57),
            (107, 58),
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
            (108, 28),
            (108, 29),
            (108, 30),
            (108, 31),
            (108, 32),
            (108, 33),
            (108, 34),
            (108, 35),
            (108, 36),
            (108, 37),
            (108, 38),
            (108, 39),
            (108, 40),
            (108, 41),
            (108, 42),
            (108, 43),
            (108, 44),
            (108, 45),
            (108, 46),
            (108, 47),
            (108, 48),
            (108, 49),
            (108, 50),
            (108, 51),
            (108, 52),
            (108, 53),
            (108, 54),
            (108, 55),
            (108, 56),
            (108, 57),
            (108, 58),
            (108, 59),
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
            (109, 29),
            (109, 30),
            (109, 31),
            (109, 32),
            (109, 33),
            (109, 34),
            (109, 35),
            (109, 36),
            (109, 37),
            (109, 38),
            (109, 39),
            (109, 40),
            (109, 41),
            (109, 42),
            (109, 43),
            (109, 44),
            (109, 45),
            (109, 46),
            (109, 47),
            (109, 48),
            (109, 49),
            (109, 50),
            (109, 51),
            (109, 52),
            (109, 53),
            (109, 54),
            (109, 55),
            (109, 56),
            (109, 57),
            (109, 58),
            (109, 59),
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
            (110, 29),
            (110, 30),
            (110, 31),
            (110, 32),
            (110, 33),
            (110, 34),
            (110, 35),
            (110, 36),
            (110, 37),
            (110, 38),
            (110, 39),
            (110, 40),
            (110, 41),
            (110, 42),
            (110, 43),
            (110, 44),
            (110, 45),
            (110, 46),
            (110, 47),
            (110, 48),
            (110, 49),
            (110, 50),
            (110, 51),
            (110, 52),
            (110, 53),
            (110, 54),
            (110, 55),
            (110, 56),
            (110, 57),
            (110, 58),
            (110, 59),
            (110, 60),
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
            (111, 29),
            (111, 30),
            (111, 31),
            (111, 32),
            (111, 33),
            (111, 34),
            (111, 35),
            (111, 36),
            (111, 37),
            (111, 38),
            (111, 39),
            (111, 40),
            (111, 41),
            (111, 42),
            (111, 43),
            (111, 44),
            (111, 45),
            (111, 46),
            (111, 47),
            (111, 48),
            (111, 49),
            (111, 50),
            (111, 51),
            (111, 52),
            (111, 53),
            (111, 54),
            (111, 55),
            (111, 56),
            (111, 57),
            (111, 58),
            (111, 59),
            (111, 60),
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
            (112, 29),
            (112, 30),
            (112, 31),
            (112, 32),
            (112, 33),
            (112, 34),
            (112, 35),
            (112, 36),
            (112, 37),
            (112, 38),
            (112, 39),
            (112, 40),
            (112, 41),
            (112, 42),
            (112, 43),
            (112, 44),
            (112, 45),
            (112, 46),
            (112, 47),
            (112, 48),
            (112, 49),
            (112, 50),
            (112, 51),
            (112, 52),
            (112, 53),
            (112, 54),
            (112, 55),
            (112, 56),
            (112, 57),
            (112, 58),
            (112, 59),
            (112, 60),
            (112, 61),
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
            (113, 30),
            (113, 31),
            (113, 32),
            (113, 33),
            (113, 34),
            (113, 35),
            (113, 36),
            (113, 37),
            (113, 38),
            (113, 39),
            (113, 40),
            (113, 41),
            (113, 42),
            (113, 43),
            (113, 44),
            (113, 45),
            (113, 46),
            (113, 47),
            (113, 48),
            (113, 49),
            (113, 50),
            (113, 51),
            (113, 52),
            (113, 53),
            (113, 54),
            (113, 55),
            (113, 56),
            (113, 57),
            (113, 58),
            (113, 59),
            (113, 60),
            (113, 61),
            (113, 62),
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
            (114, 30),
            (114, 31),
            (114, 32),
            (114, 33),
            (114, 34),
            (114, 35),
            (114, 36),
            (114, 37),
            (114, 38),
            (114, 39),
            (114, 40),
            (114, 41),
            (114, 42),
            (114, 43),
            (114, 44),
            (114, 45),
            (114, 46),
            (114, 47),
            (114, 48),
            (114, 49),
            (114, 50),
            (114, 51),
            (114, 52),
            (114, 53),
            (114, 54),
            (114, 55),
            (114, 56),
            (114, 57),
            (114, 58),
            (114, 59),
            (114, 60),
            (114, 61),
            (114, 62),
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
            (115, 30),
            (115, 31),
            (115, 32),
            (115, 33),
            (115, 34),
            (115, 35),
            (115, 36),
            (115, 37),
            (115, 38),
            (115, 39),
            (115, 40),
            (115, 41),
            (115, 42),
            (115, 43),
            (115, 44),
            (115, 45),
            (115, 46),
            (115, 47),
            (115, 48),
            (115, 49),
            (115, 50),
            (115, 51),
            (115, 52),
            (115, 53),
            (115, 54),
            (115, 55),
            (115, 56),
            (115, 57),
            (115, 58),
            (115, 59),
            (115, 60),
            (115, 61),
            (115, 62),
            (115, 63),
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
            (116, 31),
            (116, 32),
            (116, 33),
            (116, 34),
            (116, 35),
            (116, 36),
            (116, 37),
            (116, 38),
            (116, 39),
            (116, 40),
            (116, 41),
            (116, 42),
            (116, 43),
            (116, 44),
            (116, 45),
            (116, 46),
            (116, 47),
            (116, 48),
            (116, 49),
            (116, 50),
            (116, 51),
            (116, 52),
            (116, 53),
            (116, 54),
            (116, 55),
            (116, 56),
            (116, 57),
            (116, 58),
            (116, 59),
            (116, 60),
            (116, 61),
            (116, 62),
            (116, 63),
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
            (117, 31),
            (117, 32),
            (117, 33),
            (117, 34),
            (117, 35),
            (117, 36),
            (117, 37),
            (117, 38),
            (117, 39),
            (117, 40),
            (117, 41),
            (117, 42),
            (117, 43),
            (117, 44),
            (117, 45),
            (117, 46),
            (117, 47),
            (117, 48),
            (117, 49),
            (117, 50),
            (117, 51),
            (117, 52),
            (117, 53),
            (117, 54),
            (117, 55),
            (117, 56),
            (117, 57),
            (117, 58),
            (117, 59),
            (117, 60),
            (117, 61),
            (117, 62),
            (117, 63),
            (117, 64),
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
            (118, 31),
            (118, 32),
            (118, 33),
            (118, 34),
            (118, 35),
            (118, 36),
            (118, 37),
            (118, 38),
            (118, 39),
            (118, 40),
            (118, 41),
            (118, 42),
            (118, 43),
            (118, 44),
            (118, 45),
            (118, 46),
            (118, 47),
            (118, 48),
            (118, 49),
            (118, 50),
            (118, 51),
            (118, 52),
            (118, 53),
            (118, 54),
            (118, 55),
            (118, 56),
            (118, 57),
            (118, 58),
            (118, 59),
            (118, 60),
            (118, 61),
            (118, 62),
            (118, 63),
            (118, 64),
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
            (119, 31),
            (119, 32),
            (119, 33),
            (119, 34),
            (119, 35),
            (119, 36),
            (119, 37),
            (119, 38),
            (119, 39),
            (119, 40),
            (119, 41),
            (119, 42),
            (119, 43),
            (119, 44),
            (119, 45),
            (119, 46),
            (119, 47),
            (119, 48),
            (119, 49),
            (119, 50),
            (119, 51),
            (119, 52),
            (119, 53),
            (119, 54),
            (119, 55),
            (119, 56),
            (119, 57),
            (119, 58),
            (119, 59),
            (119, 60),
            (119, 61),
            (119, 62),
            (119, 63),
            (119, 64),
            (119, 65),
        }:
            return 5
        elif key in {
            (2, 0),
            (4, 1),
            (7, 3),
            (9, 4),
            (11, 5),
            (13, 6),
            (14, 7),
            (16, 8),
            (18, 9),
            (20, 10),
            (23, 12),
            (25, 13),
            (27, 14),
            (29, 15),
            (32, 17),
            (34, 18),
            (36, 19),
            (39, 21),
            (41, 22),
            (43, 23),
            (45, 24),
            (48, 26),
            (50, 27),
            (52, 28),
            (55, 30),
            (57, 31),
            (59, 32),
            (61, 33),
            (64, 35),
            (66, 36),
            (68, 37),
            (70, 38),
            (73, 40),
            (75, 41),
            (77, 42),
            (80, 44),
            (82, 45),
            (84, 46),
            (86, 47),
            (89, 49),
            (91, 50),
            (93, 51),
            (96, 53),
            (98, 54),
            (100, 55),
            (102, 56),
            (105, 58),
            (107, 59),
            (109, 60),
            (111, 61),
            (114, 63),
            (116, 64),
            (118, 65),
        }:
            return 7
        return 24

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_3_output, num_attn_1_7_output):
        key = (num_attn_2_3_output, num_attn_1_7_output)
        return 16

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_1_output, num_attn_1_0_output):
        key = (num_attn_2_1_output, num_attn_1_0_output)
        return 27

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_0_outputs)
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


print(run(["<s>", "5", "0", "3", "3", "3", "1", "3", "5", "2", "4", "0", "0", "4"]))
