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
        "output/length/rasp/mostfreq/trainlength30/s1/most_freq_weights.csv",
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
        if position in {0}:
            return token == "2"
        elif position in {1, 7, 19, 23, 29}:
            return token == "4"
        elif position in {2, 3}:
            return token == "1"
        elif position in {4, 5, 6, 11, 13, 14, 21, 22, 24, 25, 26, 28}:
            return token == "3"
        elif position in {8, 9, 36}:
            return token == "5"
        elif position in {10}:
            return token == "0"
        elif position in {32, 33, 34, 35, 37, 38, 39, 12, 15, 16, 17, 18, 20, 30, 31}:
            return token == ""
        elif position in {27}:
            return token == "<s>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 2
        elif q_position in {1, 13}:
            return k_position == 1
        elif q_position in {3, 7}:
            return k_position == 4
        elif q_position in {8, 11, 4}:
            return k_position == 5
        elif q_position in {18, 12, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 14
        elif q_position in {9, 10, 19}:
            return k_position == 3
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 11
        elif q_position in {16, 17, 32}:
            return k_position == 8
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {33, 35, 21}:
            return k_position == 32
        elif q_position in {22}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 15
        elif q_position in {24}:
            return k_position == 26
        elif q_position in {25, 27}:
            return k_position == 24
        elif q_position in {26, 37, 38}:
            return k_position == 9
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {34, 29}:
            return k_position == 17
        elif q_position in {30}:
            return k_position == 12
        elif q_position in {31}:
            return k_position == 39
        elif q_position in {36}:
            return k_position == 30
        elif q_position in {39}:
            return k_position == 37

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 7, 9, 10, 15, 16, 18, 19, 20, 22, 23, 25, 27, 29}:
            return token == "4"
        elif position in {1, 4}:
            return token == "5"
        elif position in {17, 2, 3, 35}:
            return token == "3"
        elif position in {5, 6}:
            return token == "1"
        elif position in {8, 12, 13, 21, 24, 26, 28}:
            return token == "2"
        elif position in {11}:
            return token == "0"
        elif position in {32, 33, 34, 36, 37, 38, 39, 14, 30, 31}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 7, 9, 16, 22, 23}:
            return token == "4"
        elif position in {1, 5, 6, 10, 13, 14, 19, 25}:
            return token == "0"
        elif position in {2, 12, 15}:
            return token == "1"
        elif position in {17, 3}:
            return token == "3"
        elif position in {11, 4}:
            return token == "2"
        elif position in {
            32,
            34,
            35,
            36,
            37,
            38,
            39,
            8,
            18,
            20,
            21,
            24,
            26,
            27,
            28,
            29,
            30,
            31,
        }:
            return token == ""
        elif position in {33}:
            return token == "<s>"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_token, k_token):
        if q_token in {"1", "0", "5", "2", "3", "4"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "2"

    attn_0_4_pattern = select_closest(tokens, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 10, 21, 6}:
            return token == "0"
        elif position in {1, 11, 9}:
            return token == "1"
        elif position in {2, 3, 4}:
            return token == "5"
        elif position in {17, 5}:
            return token == "3"
        elif position in {7, 14, 16, 19, 22, 23, 25, 27, 29}:
            return token == "4"
        elif position in {8, 12, 13, 15, 20, 26}:
            return token == "2"
        elif position in {32, 33, 34, 35, 36, 37, 38, 39, 18, 28, 30, 31}:
            return token == ""
        elif position in {24}:
            return token == "<s>"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1, 4, 37, 5, 20}:
            return token == "5"
        elif position in {2, 3, 6, 7, 9, 10, 12, 14, 15, 16, 30}:
            return token == "0"
        elif position in {8}:
            return token == "1"
        elif position in {
            11,
            13,
            17,
            18,
            19,
            21,
            22,
            24,
            25,
            26,
            27,
            28,
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
        elif position in {29, 23}:
            return token == "4"

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 35
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 10, 4}:
            return k_position == 2
        elif q_position in {3, 7, 9, 13, 19, 22, 29}:
            return k_position == 5
        elif q_position in {20, 5}:
            return k_position == 3
        elif q_position in {11, 12, 6}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {37, 14}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16, 17, 27, 38}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 21
        elif q_position in {35, 21}:
            return k_position == 8
        elif q_position in {23}:
            return k_position == 23
        elif q_position in {24}:
            return k_position == 24
        elif q_position in {25, 39, 33}:
            return k_position == 11
        elif q_position in {26}:
            return k_position == 14
        elif q_position in {28}:
            return k_position == 28
        elif q_position in {30}:
            return k_position == 15
        elif q_position in {31}:
            return k_position == 16
        elif q_position in {32, 36}:
            return k_position == 20
        elif q_position in {34}:
            return k_position == 13

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
            3,
            4,
            5,
            6,
            12,
            15,
            16,
            19,
            20,
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
            return token == ""
        elif position in {1}:
            return token == "3"
        elif position in {9, 11, 7}:
            return token == "4"
        elif position in {8, 10, 13, 14, 17, 21, 22, 24, 25}:
            return token == "<s>"
        elif position in {18}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 5}:
            return token == "<s>"
        elif position in {1, 2, 3, 4}:
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
        }:
            return token == ""
        elif position in {26, 15}:
            return token == "<pad>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 1, 8, 7}:
            return token == "5"
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
        }:
            return token == ""
        elif position in {6}:
            return token == "<s>"
        elif position in {17, 25}:
            return token == "<pad>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 5, 6}:
            return token == "2"
        elif position in {1}:
            return token == "4"
        elif position in {
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

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 1}:
            return token == "4"
        elif position in {2, 3, 4}:
            return token == "<s>"
        elif position in {5, 6}:
            return token == "3"
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
        elif position in {20}:
            return token == "<pad>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 1, 2, 3, 4, 5, 6, 32, 33, 34, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {7, 10, 14, 17, 19, 20, 21, 22, 24, 25, 27}:
            return token == "3"
        elif position in {8, 9, 11, 12, 13, 18, 29}:
            return token == "4"
        elif position in {16, 26, 28, 15}:
            return token == "5"
        elif position in {23}:
            return token == "0"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 1}:
            return token == "2"
        elif position in {
            2,
            3,
            4,
            5,
            6,
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
        }:
            return token == ""
        elif position in {9, 7}:
            return token == "<s>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 10, 5, 6}:
            return token == "0"
        elif position in {
            1,
            2,
            3,
            4,
            8,
            12,
            13,
            15,
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
        elif position in {16, 7}:
            return token == "<s>"
        elif position in {9, 11, 14}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, position):
        key = (attn_0_2_output, position)
        if key in {
            ("0", 3),
            ("0", 4),
            ("1", 0),
            ("1", 1),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("1", 13),
            ("1", 14),
            ("1", 15),
            ("1", 19),
            ("1", 23),
            ("1", 29),
            ("1", 30),
            ("1", 31),
            ("1", 32),
            ("1", 33),
            ("1", 34),
            ("1", 35),
            ("1", 36),
            ("1", 37),
            ("1", 38),
            ("1", 39),
            ("2", 3),
            ("3", 3),
            ("3", 4),
            ("4", 3),
            ("4", 4),
            ("5", 3),
            ("5", 4),
        }:
            return 2
        elif key in {
            ("0", 8),
            ("0", 19),
            ("2", 19),
            ("3", 8),
            ("3", 19),
            ("4", 9),
            ("4", 11),
            ("4", 12),
            ("4", 14),
            ("4", 15),
            ("4", 16),
            ("4", 17),
            ("4", 18),
            ("4", 19),
            ("4", 20),
            ("4", 21),
            ("4", 22),
            ("4", 23),
            ("4", 24),
            ("4", 25),
            ("4", 26),
            ("4", 28),
            ("4", 29),
            ("4", 31),
            ("4", 36),
            ("5", 8),
            ("5", 19),
            ("<s>", 19),
        }:
            return 15
        elif key in {
            ("0", 5),
            ("0", 6),
            ("2", 5),
            ("2", 6),
            ("3", 5),
            ("3", 6),
            ("5", 5),
            ("5", 6),
            ("<s>", 5),
            ("<s>", 6),
        }:
            return 12
        elif key in {("0", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1), ("<s>", 1)}:
            return 29
        elif key in {
            ("2", 4),
            ("2", 8),
            ("4", 0),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 8),
            ("4", 13),
            ("4", 27),
            ("4", 30),
            ("4", 32),
            ("4", 33),
            ("4", 34),
            ("4", 35),
            ("4", 37),
            ("4", 38),
            ("4", 39),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 8),
            ("<s>", 10),
        }:
            return 6
        elif key in {("0", 2), ("2", 2), ("3", 2), ("4", 2), ("5", 2), ("<s>", 2)}:
            return 36
        elif key in {("<s>", 0), ("<s>", 7)}:
            return 35
        return 3

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_4_output):
        key = (position, attn_0_4_output)
        if key in {
            (0, "<s>"),
            (2, "<s>"),
            (3, "<s>"),
            (4, "<s>"),
            (5, "<s>"),
            (6, "<s>"),
            (7, "<s>"),
            (8, "<s>"),
            (9, "<s>"),
            (10, "<s>"),
            (11, "1"),
            (11, "2"),
            (11, "3"),
            (11, "5"),
            (11, "<s>"),
            (12, "1"),
            (12, "2"),
            (12, "3"),
            (12, "4"),
            (12, "5"),
            (12, "<s>"),
            (13, "1"),
            (13, "2"),
            (13, "3"),
            (13, "4"),
            (13, "5"),
            (13, "<s>"),
            (14, "<s>"),
            (15, "1"),
            (15, "2"),
            (15, "3"),
            (15, "4"),
            (15, "5"),
            (15, "<s>"),
            (16, "<s>"),
            (17, "<s>"),
            (18, "<s>"),
            (19, "<s>"),
            (20, "<s>"),
            (21, "1"),
            (21, "2"),
            (21, "3"),
            (21, "4"),
            (21, "5"),
            (21, "<s>"),
            (22, "1"),
            (22, "2"),
            (22, "3"),
            (22, "4"),
            (22, "5"),
            (22, "<s>"),
            (23, "<s>"),
            (24, "1"),
            (24, "2"),
            (24, "3"),
            (24, "4"),
            (24, "5"),
            (24, "<s>"),
            (25, "<s>"),
            (26, "3"),
            (26, "5"),
            (26, "<s>"),
            (27, "3"),
            (27, "<s>"),
            (28, "<s>"),
            (29, "<s>"),
            (30, "<s>"),
            (31, "<s>"),
            (32, "<s>"),
            (33, "<s>"),
            (34, "<s>"),
            (35, "<s>"),
            (36, "<s>"),
            (37, "<s>"),
            (38, "<s>"),
            (39, "<s>"),
        }:
            return 27
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (1, "3"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (7, "3"),
            (9, "0"),
            (9, "1"),
            (9, "2"),
            (9, "3"),
            (9, "4"),
            (9, "5"),
            (10, "3"),
            (14, "3"),
            (15, "0"),
            (16, "3"),
            (16, "4"),
            (17, "3"),
            (18, "0"),
            (18, "1"),
            (18, "2"),
            (18, "3"),
            (18, "4"),
            (18, "5"),
            (19, "0"),
            (19, "1"),
            (19, "2"),
            (19, "3"),
            (19, "4"),
            (19, "5"),
            (20, "3"),
            (25, "3"),
            (30, "3"),
            (30, "4"),
            (31, "3"),
            (31, "4"),
            (32, "3"),
            (32, "4"),
            (33, "3"),
            (33, "4"),
            (34, "3"),
            (34, "4"),
            (35, "3"),
            (35, "4"),
            (36, "3"),
            (37, "3"),
            (37, "4"),
            (38, "3"),
            (38, "4"),
            (39, "3"),
            (39, "4"),
        }:
            return 3
        elif key in {
            (1, "1"),
            (7, "1"),
            (8, "1"),
            (10, "1"),
            (16, "1"),
            (17, "1"),
            (23, "1"),
            (23, "2"),
            (26, "1"),
            (28, "1"),
            (28, "2"),
            (28, "3"),
            (28, "4"),
            (28, "5"),
            (29, "1"),
            (30, "1"),
            (31, "1"),
            (32, "1"),
            (33, "1"),
            (34, "1"),
            (35, "1"),
            (36, "1"),
            (37, "1"),
            (38, "1"),
            (39, "1"),
        }:
            return 14
        return 9

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_4_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_6_output):
        key = attn_0_6_output
        if key in {""}:
            return 10
        elif key in {""}:
            return 25
        elif key in {""}:
            return 17
        return 12

    mlp_0_2_outputs = [mlp_0_2(k0) for k0 in attn_0_6_outputs]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_6_output, position):
        key = (attn_0_6_output, position)
        if key in {
            ("0", 3),
            ("0", 4),
            ("0", 10),
            ("0", 12),
            ("0", 16),
            ("0", 17),
            ("0", 18),
            ("0", 21),
            ("0", 22),
            ("0", 24),
            ("0", 26),
            ("0", 27),
            ("0", 28),
            ("1", 3),
            ("1", 17),
            ("1", 18),
            ("1", 21),
            ("1", 22),
            ("1", 24),
            ("1", 26),
            ("1", 28),
            ("2", 3),
            ("2", 16),
            ("2", 17),
            ("2", 18),
            ("2", 21),
            ("2", 22),
            ("2", 24),
            ("2", 26),
            ("2", 27),
            ("2", 28),
            ("3", 3),
            ("3", 17),
            ("3", 18),
            ("3", 20),
            ("3", 21),
            ("3", 22),
            ("3", 24),
            ("3", 26),
            ("3", 28),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 7),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 16),
            ("4", 17),
            ("4", 18),
            ("4", 19),
            ("4", 20),
            ("4", 21),
            ("4", 22),
            ("4", 23),
            ("4", 24),
            ("4", 26),
            ("4", 27),
            ("4", 28),
            ("4", 29),
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
            ("<s>", 3),
            ("<s>", 10),
            ("<s>", 12),
            ("<s>", 16),
            ("<s>", 17),
            ("<s>", 18),
            ("<s>", 21),
            ("<s>", 22),
            ("<s>", 24),
            ("<s>", 26),
            ("<s>", 27),
            ("<s>", 28),
            ("<s>", 39),
        }:
            return 33
        elif key in {
            ("0", 6),
            ("0", 8),
            ("0", 9),
            ("0", 13),
            ("0", 14),
            ("0", 23),
            ("1", 6),
            ("1", 9),
            ("1", 13),
            ("1", 14),
            ("2", 6),
            ("2", 9),
            ("2", 13),
            ("2", 14),
            ("3", 6),
            ("3", 9),
            ("3", 13),
            ("3", 14),
            ("4", 6),
            ("4", 9),
            ("4", 13),
            ("4", 14),
            ("5", 6),
            ("5", 13),
            ("<s>", 6),
            ("<s>", 9),
            ("<s>", 13),
            ("<s>", 14),
        }:
            return 24
        elif key in {
            ("0", 25),
            ("1", 25),
            ("2", 25),
            ("3", 25),
            ("4", 25),
            ("5", 0),
            ("5", 1),
            ("5", 2),
            ("5", 3),
            ("5", 4),
            ("5", 5),
            ("5", 7),
            ("5", 8),
            ("5", 9),
            ("5", 10),
            ("5", 11),
            ("5", 12),
            ("5", 14),
            ("5", 15),
            ("5", 16),
            ("5", 17),
            ("5", 18),
            ("5", 19),
            ("5", 20),
            ("5", 21),
            ("5", 22),
            ("5", 23),
            ("5", 24),
            ("5", 25),
            ("5", 26),
            ("5", 27),
            ("5", 28),
            ("5", 29),
            ("5", 30),
            ("5", 31),
            ("5", 32),
            ("5", 33),
            ("5", 34),
            ("5", 35),
            ("5", 36),
            ("5", 37),
            ("5", 38),
            ("5", 39),
            ("<s>", 25),
        }:
            return 19
        elif key in {
            ("1", 8),
            ("1", 20),
            ("1", 23),
            ("2", 8),
            ("2", 20),
            ("2", 23),
            ("3", 8),
            ("4", 8),
            ("<s>", 8),
            ("<s>", 20),
            ("<s>", 23),
        }:
            return 10
        elif key in {
            ("0", 1),
            ("0", 2),
            ("1", 1),
            ("1", 2),
            ("1", 15),
            ("2", 1),
            ("2", 2),
            ("2", 30),
            ("2", 31),
            ("2", 32),
            ("2", 34),
            ("2", 35),
            ("2", 36),
            ("2", 38),
            ("3", 1),
            ("3", 2),
            ("4", 1),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 11
        elif key in {
            ("0", 15),
            ("2", 4),
            ("2", 10),
            ("2", 12),
            ("2", 15),
            ("2", 33),
            ("2", 37),
            ("2", 39),
            ("3", 15),
            ("4", 15),
            ("<s>", 4),
            ("<s>", 15),
        }:
            return 5
        return 4

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_6_outputs, positions)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 12

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_5_output, num_attn_0_4_output):
        key = (num_attn_0_5_output, num_attn_0_4_output)
        return 5

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_5_output):
        key = num_attn_0_5_output
        return 2

    num_mlp_0_2_outputs = [num_mlp_0_2(k0) for k0 in num_attn_0_5_outputs]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_6_output):
        key = num_attn_0_6_output
        return 6

    num_mlp_0_3_outputs = [num_mlp_0_3(k0) for k0 in num_attn_0_6_outputs]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, mlp_0_3_output):
        if position in {0, 2}:
            return mlp_0_3_output == 18
        elif position in {1, 10, 7}:
            return mlp_0_3_output == 8
        elif position in {32, 33, 34, 3, 4, 5, 6, 35, 37, 39, 30, 31}:
            return mlp_0_3_output == 24
        elif position in {8, 9, 12, 15, 17, 26}:
            return mlp_0_3_output == 19
        elif position in {11}:
            return mlp_0_3_output == 28
        elif position in {13, 14, 18, 19, 21, 27}:
            return mlp_0_3_output == 4
        elif position in {16, 25, 22}:
            return mlp_0_3_output == 11
        elif position in {20, 29}:
            return mlp_0_3_output == 36
        elif position in {23}:
            return mlp_0_3_output == 5
        elif position in {24}:
            return mlp_0_3_output == 32
        elif position in {28}:
            return mlp_0_3_output == 10
        elif position in {36, 38}:
            return mlp_0_3_output == 6

    attn_1_0_pattern = select_closest(mlp_0_3_outputs, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, mlp_0_3_output):
        if position in {0, 2, 3, 5, 27, 29}:
            return mlp_0_3_output == 18
        elif position in {1}:
            return mlp_0_3_output == 32
        elif position in {4}:
            return mlp_0_3_output == 0
        elif position in {20, 6}:
            return mlp_0_3_output == 11
        elif position in {7}:
            return mlp_0_3_output == 15
        elif position in {8, 26, 35}:
            return mlp_0_3_output == 10
        elif position in {9}:
            return mlp_0_3_output == 30
        elif position in {10, 11, 14}:
            return mlp_0_3_output == 4
        elif position in {12}:
            return mlp_0_3_output == 28
        elif position in {13}:
            return mlp_0_3_output == 1
        elif position in {15}:
            return mlp_0_3_output == 33
        elif position in {16, 33, 31}:
            return mlp_0_3_output == 9
        elif position in {32, 17, 18, 22, 23, 25, 28}:
            return mlp_0_3_output == 37
        elif position in {19}:
            return mlp_0_3_output == 5
        elif position in {24, 21}:
            return mlp_0_3_output == 14
        elif position in {34, 36, 30, 39}:
            return mlp_0_3_output == 24
        elif position in {37, 38}:
            return mlp_0_3_output == 13

    attn_1_1_pattern = select_closest(mlp_0_3_outputs, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_0_output, attn_0_2_output):
        if attn_0_0_output in {"1", "0", "2", "3", "4"}:
            return attn_0_2_output == ""
        elif attn_0_0_output in {"5"}:
            return attn_0_2_output == "4"
        elif attn_0_0_output in {"<s>"}:
            return attn_0_2_output == "<pad>"

    attn_1_2_pattern = select_closest(attn_0_2_outputs, attn_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, mlp_0_3_output):
        if token in {"0"}:
            return mlp_0_3_output == 16
        elif token in {"5", "1", "4"}:
            return mlp_0_3_output == 18
        elif token in {"2"}:
            return mlp_0_3_output == 24
        elif token in {"3"}:
            return mlp_0_3_output == 7
        elif token in {"<s>"}:
            return mlp_0_3_output == 19

    attn_1_3_pattern = select_closest(mlp_0_3_outputs, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(token, mlp_0_0_output):
        if token in {"0"}:
            return mlp_0_0_output == 29
        elif token in {"1"}:
            return mlp_0_0_output == 35
        elif token in {"2", "3"}:
            return mlp_0_0_output == 24
        elif token in {"4"}:
            return mlp_0_0_output == 36
        elif token in {"5", "<s>"}:
            return mlp_0_0_output == 34

    attn_1_4_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, mlp_0_1_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0, 38}:
            return k_position == 28
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {2}:
            return k_position == 19
        elif q_position in {32, 35, 3, 39, 30}:
            return k_position == 24
        elif q_position in {4, 5, 6}:
            return k_position == 7
        elif q_position in {7, 9, 10, 11, 14, 16, 18, 20, 24}:
            return k_position == 22
        elif q_position in {8, 31}:
            return k_position == 20
        elif q_position in {12, 13}:
            return k_position == 27
        elif q_position in {15}:
            return k_position == 9
        elif q_position in {17, 27}:
            return k_position == 23
        elif q_position in {34, 19, 21, 22, 23, 28, 29}:
            return k_position == 15
        elif q_position in {25}:
            return k_position == 13
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {33, 36}:
            return k_position == 5
        elif q_position in {37}:
            return k_position == 32

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, mlp_0_0_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_1_output, mlp_0_3_output):
        if attn_0_1_output in {"0"}:
            return mlp_0_3_output == 10
        elif attn_0_1_output in {"1"}:
            return mlp_0_3_output == 25
        elif attn_0_1_output in {"2", "4"}:
            return mlp_0_3_output == 14
        elif attn_0_1_output in {"3"}:
            return mlp_0_3_output == 37
        elif attn_0_1_output in {"5"}:
            return mlp_0_3_output == 18
        elif attn_0_1_output in {"<s>"}:
            return mlp_0_3_output == 19

    attn_1_6_pattern = select_closest(mlp_0_3_outputs, attn_0_1_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_1_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_4_output, token):
        if attn_0_4_output in {"<s>", "1", "0", "5", "3", "4"}:
            return token == "<s>"
        elif attn_0_4_output in {"2"}:
            return token == "5"

    attn_1_7_pattern = select_closest(tokens, attn_0_4_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, mlp_0_1_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_0_output, attn_0_6_output):
        if mlp_0_0_output in {0, 4, 6, 7, 8, 39, 16, 19, 25, 26}:
            return attn_0_6_output == "<pad>"
        elif mlp_0_0_output in {
            32,
            1,
            2,
            3,
            34,
            35,
            37,
            38,
            9,
            13,
            17,
            21,
            22,
            24,
            27,
            31,
        }:
            return attn_0_6_output == ""
        elif mlp_0_0_output in {5, 23}:
            return attn_0_6_output == "5"
        elif mlp_0_0_output in {10, 18, 28, 30}:
            return attn_0_6_output == "3"
        elif mlp_0_0_output in {11, 36, 29}:
            return attn_0_6_output == "2"
        elif mlp_0_0_output in {12, 20}:
            return attn_0_6_output == "1"
        elif mlp_0_0_output in {33, 14}:
            return attn_0_6_output == "4"
        elif mlp_0_0_output in {15}:
            return attn_0_6_output == "<s>"

    num_attn_1_0_pattern = select(attn_0_6_outputs, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_6_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_3_output, attn_0_5_output):
        if mlp_0_3_output in {0, 1, 33, 4, 36, 39, 8, 11, 13, 14, 29}:
            return attn_0_5_output == "5"
        elif mlp_0_3_output in {2, 3, 31}:
            return attn_0_5_output == "<s>"
        elif mlp_0_3_output in {
            5,
            6,
            7,
            9,
            10,
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
            35,
            38,
        }:
            return attn_0_5_output == ""
        elif mlp_0_3_output in {12}:
            return attn_0_5_output == "1"
        elif mlp_0_3_output in {32}:
            return attn_0_5_output == "2"
        elif mlp_0_3_output in {34, 37}:
            return attn_0_5_output == "4"

    num_attn_1_1_pattern = select(attn_0_5_outputs, mlp_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_4_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_0_output, attn_0_5_output):
        if mlp_0_0_output in {0, 8, 37, 23}:
            return attn_0_5_output == "4"
        elif mlp_0_0_output in {1, 11, 29}:
            return attn_0_5_output == "0"
        elif mlp_0_0_output in {
            2,
            3,
            6,
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
            24,
            26,
            30,
            32,
            34,
            35,
            36,
            38,
        }:
            return attn_0_5_output == ""
        elif mlp_0_0_output in {4, 28}:
            return attn_0_5_output == "<pad>"
        elif mlp_0_0_output in {33, 27, 5, 39}:
            return attn_0_5_output == "1"
        elif mlp_0_0_output in {18, 31, 7}:
            return attn_0_5_output == "5"
        elif mlp_0_0_output in {9}:
            return attn_0_5_output == "<s>"
        elif mlp_0_0_output in {25}:
            return attn_0_5_output == "2"

    num_attn_1_2_pattern = select(attn_0_5_outputs, mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_7_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(mlp_0_0_output, attn_0_3_output):
        if mlp_0_0_output in {
            0,
            1,
            3,
            6,
            8,
            11,
            12,
            15,
            16,
            18,
            19,
            21,
            22,
            26,
            27,
            28,
            29,
            31,
            33,
            36,
            37,
            39,
        }:
            return attn_0_3_output == ""
        elif mlp_0_0_output in {32, 2, 34, 4, 35, 38, 10, 13, 14, 17, 24, 30}:
            return attn_0_3_output == "1"
        elif mlp_0_0_output in {5}:
            return attn_0_3_output == "3"
        elif mlp_0_0_output in {9, 20, 7}:
            return attn_0_3_output == "<s>"
        elif mlp_0_0_output in {25, 23}:
            return attn_0_3_output == "5"

    num_attn_1_3_pattern = select(attn_0_3_outputs, mlp_0_0_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, attn_0_2_output):
        if position in {0, 33, 31}:
            return attn_0_2_output == "5"
        elif position in {1, 36, 37, 29, 30}:
            return attn_0_2_output == "4"
        elif position in {2, 3, 4}:
            return attn_0_2_output == "3"
        elif position in {
            5,
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            18,
            19,
            22,
            23,
            24,
            25,
            26,
            28,
            32,
            34,
            39,
        }:
            return attn_0_2_output == ""
        elif position in {8, 15, 17, 20, 21, 27}:
            return attn_0_2_output == "<pad>"
        elif position in {35}:
            return attn_0_2_output == "<s>"
        elif position in {38}:
            return attn_0_2_output == "1"

    num_attn_1_4_pattern = select(attn_0_2_outputs, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_4_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {"<s>", "1", "0", "5", "3", "4"}:
            return attn_0_1_output == ""
        elif attn_0_3_output in {"2"}:
            return attn_0_1_output == "2"

    num_attn_1_5_pattern = select(attn_0_1_outputs, attn_0_3_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_6_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(mlp_0_3_output, attn_0_7_output):
        if mlp_0_3_output in {
            0,
            1,
            2,
            4,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            15,
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
            return attn_0_7_output == ""
        elif mlp_0_3_output in {3}:
            return attn_0_7_output == "<s>"
        elif mlp_0_3_output in {5, 14, 16, 18, 24}:
            return attn_0_7_output == "4"
        elif mlp_0_3_output in {30}:
            return attn_0_7_output == "<pad>"

    num_attn_1_6_pattern = select(attn_0_7_outputs, mlp_0_3_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_4_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_5_output, attn_0_3_output):
        if attn_0_5_output in {"<s>", "1", "0", "2", "3"}:
            return attn_0_3_output == ""
        elif attn_0_5_output in {"4"}:
            return attn_0_3_output == "4"
        elif attn_0_5_output in {"5"}:
            return attn_0_3_output == "<pad>"

    num_attn_1_7_pattern = select(attn_0_3_outputs, attn_0_5_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output, mlp_0_1_output):
        key = (attn_0_0_output, mlp_0_1_output)
        return 14

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, mlp_0_1_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_5_output, mlp_0_0_output):
        key = (attn_0_5_output, mlp_0_0_output)
        return 8

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_5_outputs, mlp_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_0_1_output, attn_0_6_output):
        key = (attn_0_1_output, attn_0_6_output)
        return 8

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_6_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(position):
        key = position
        return 15

    mlp_1_3_outputs = [mlp_1_3(k0) for k0 in positions]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_1_7_output):
        key = (num_attn_1_2_output, num_attn_1_7_output)
        return 24

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        if key in {0}:
            return 14
        return 10

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_2_output, num_attn_1_6_output):
        key = (num_attn_1_2_output, num_attn_1_6_output)
        return 6

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_5_output, num_attn_1_6_output):
        key = (num_attn_0_5_output, num_attn_1_6_output)
        return 5

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"0"}:
            return mlp_0_0_output == 29
        elif attn_0_1_output in {"1"}:
            return mlp_0_0_output == 12
        elif attn_0_1_output in {"5", "<s>", "2", "3"}:
            return mlp_0_0_output == 35
        elif attn_0_1_output in {"4"}:
            return mlp_0_0_output == 3

    attn_2_0_pattern = select_closest(mlp_0_0_outputs, attn_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_5_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, num_mlp_1_1_output):
        if token in {"0"}:
            return num_mlp_1_1_output == 34
        elif token in {"1"}:
            return num_mlp_1_1_output == 18
        elif token in {"2"}:
            return num_mlp_1_1_output == 27
        elif token in {"3"}:
            return num_mlp_1_1_output == 1
        elif token in {"4"}:
            return num_mlp_1_1_output == 28
        elif token in {"5"}:
            return num_mlp_1_1_output == 11
        elif token in {"<s>"}:
            return num_mlp_1_1_output == 24

    attn_2_1_pattern = select_closest(num_mlp_1_1_outputs, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_6_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_7_output, mlp_0_0_output):
        if attn_0_7_output in {"0"}:
            return mlp_0_0_output == 2
        elif attn_0_7_output in {"<s>", "1", "5", "2", "4"}:
            return mlp_0_0_output == 36
        elif attn_0_7_output in {"3"}:
            return mlp_0_0_output == 29

    attn_2_2_pattern = select_closest(mlp_0_0_outputs, attn_0_7_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(position, mlp_0_1_output):
        if position in {0, 2, 34, 37, 10, 15, 16, 18, 23, 26, 28, 30}:
            return mlp_0_1_output == 27
        elif position in {1}:
            return mlp_0_1_output == 38
        elif position in {3}:
            return mlp_0_1_output == 0
        elif position in {4}:
            return mlp_0_1_output == 9
        elif position in {35, 5}:
            return mlp_0_1_output == 17
        elif position in {6}:
            return mlp_0_1_output == 2
        elif position in {7, 9, 11, 12, 14, 17, 19, 20, 21, 22, 24, 25, 27, 29}:
            return mlp_0_1_output == 14
        elif position in {8, 36, 13}:
            return mlp_0_1_output == 28
        elif position in {31}:
            return mlp_0_1_output == 18
        elif position in {32}:
            return mlp_0_1_output == 26
        elif position in {33}:
            return mlp_0_1_output == 15
        elif position in {38}:
            return mlp_0_1_output == 7
        elif position in {39}:
            return mlp_0_1_output == 3

    attn_2_3_pattern = select_closest(mlp_0_1_outputs, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_4_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(token, position):
        if token in {"0"}:
            return position == 37
        elif token in {"3", "1", "4"}:
            return position == 0
        elif token in {"2"}:
            return position == 25
        elif token in {"5"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 20

    attn_2_4_pattern = select_closest(positions, tokens, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_7_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_5_output, mlp_0_3_output):
        if attn_0_5_output in {"0"}:
            return mlp_0_3_output == 14
        elif attn_0_5_output in {"1"}:
            return mlp_0_3_output == 12
        elif attn_0_5_output in {"2"}:
            return mlp_0_3_output == 0
        elif attn_0_5_output in {"3"}:
            return mlp_0_3_output == 30
        elif attn_0_5_output in {"4"}:
            return mlp_0_3_output == 8
        elif attn_0_5_output in {"5"}:
            return mlp_0_3_output == 5
        elif attn_0_5_output in {"<s>"}:
            return mlp_0_3_output == 17

    attn_2_5_pattern = select_closest(mlp_0_3_outputs, attn_0_5_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_6_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_7_output, mlp_0_0_output):
        if attn_0_7_output in {"5", "0"}:
            return mlp_0_0_output == 2
        elif attn_0_7_output in {"1"}:
            return mlp_0_0_output == 12
        elif attn_0_7_output in {"2"}:
            return mlp_0_0_output == 7
        elif attn_0_7_output in {"3"}:
            return mlp_0_0_output == 37
        elif attn_0_7_output in {"4"}:
            return mlp_0_0_output == 3
        elif attn_0_7_output in {"<s>"}:
            return mlp_0_0_output == 11

    attn_2_6_pattern = select_closest(mlp_0_0_outputs, attn_0_7_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, num_mlp_0_3_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(position, mlp_0_3_output):
        if position in {0, 33, 32, 34, 4, 35, 36, 37, 38, 39, 30}:
            return mlp_0_3_output == 24
        elif position in {1}:
            return mlp_0_3_output == 2
        elif position in {2, 27, 12, 15}:
            return mlp_0_3_output == 36
        elif position in {3}:
            return mlp_0_3_output == 18
        elif position in {5}:
            return mlp_0_3_output == 14
        elif position in {6}:
            return mlp_0_3_output == 10
        elif position in {
            7,
            8,
            9,
            11,
            13,
            14,
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
        }:
            return mlp_0_3_output == 11
        elif position in {10, 29}:
            return mlp_0_3_output == 19
        elif position in {31}:
            return mlp_0_3_output == 15

    attn_2_7_pattern = select_closest(mlp_0_3_outputs, positions, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_1_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, attn_0_3_output):
        if position in {
            0,
            1,
            2,
            3,
            4,
            7,
            9,
            10,
            11,
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
            33,
            39,
        }:
            return attn_0_3_output == ""
        elif position in {5, 6}:
            return attn_0_3_output == "4"
        elif position in {8, 18, 12}:
            return attn_0_3_output == "<pad>"
        elif position in {32, 35, 38}:
            return attn_0_3_output == "1"
        elif position in {34, 36, 37}:
            return attn_0_3_output == "3"

    num_attn_2_0_pattern = select(attn_0_3_outputs, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(q_position, k_position):
        if q_position in {0, 22}:
            return k_position == 20
        elif q_position in {1, 38, 23}:
            return k_position == 18
        elif q_position in {32, 25, 2, 14}:
            return k_position == 14
        elif q_position in {3, 4}:
            return k_position == 29
        elif q_position in {13, 5}:
            return k_position == 10
        elif q_position in {12, 6, 31}:
            return k_position == 6
        elif q_position in {20, 36, 7}:
            return k_position == 25
        elif q_position in {8}:
            return k_position == 33
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {35, 10, 15, 16, 18, 19, 21, 27}:
            return k_position == 0
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {17, 33}:
            return k_position == 19
        elif q_position in {24}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 4
        elif q_position in {28}:
            return k_position == 11
        elif q_position in {34, 29, 39}:
            return k_position == 17
        elif q_position in {30}:
            return k_position == 23
        elif q_position in {37}:
            return k_position == 16

    num_attn_2_1_pattern = select(positions, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_3_output, attn_0_7_output):
        if mlp_0_3_output in {
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
            11,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            22,
            25,
            26,
            27,
            29,
            30,
            31,
            33,
            38,
        }:
            return attn_0_7_output == ""
        elif mlp_0_3_output in {32, 37, 10, 12, 18, 24, 28}:
            return attn_0_7_output == "5"
        elif mlp_0_3_output in {36, 39, 23}:
            return attn_0_7_output == "<pad>"
        elif mlp_0_3_output in {34}:
            return attn_0_7_output == "<s>"
        elif mlp_0_3_output in {35}:
            return attn_0_7_output == "4"

    num_attn_2_2_pattern = select(attn_0_7_outputs, mlp_0_3_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_6_output, attn_0_5_output):
        if attn_1_6_output in {"2", "0"}:
            return attn_0_5_output == "3"
        elif attn_1_6_output in {"5", "<s>", "1", "4"}:
            return attn_0_5_output == ""
        elif attn_1_6_output in {"3"}:
            return attn_0_5_output == "0"

    num_attn_2_3_pattern = select(attn_0_5_outputs, attn_1_6_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(token, position):
        if token in {"3", "1", "0"}:
            return position == 7
        elif token in {"2"}:
            return position == 18
        elif token in {"4"}:
            return position == 34
        elif token in {"5"}:
            return position == 21
        elif token in {"<s>"}:
            return position == 15

    num_attn_2_4_pattern = select(positions, tokens, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_6_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_0_output, attn_0_7_output):
        if attn_0_0_output in {"1", "0", "5", "2", "4"}:
            return attn_0_7_output == ""
        elif attn_0_0_output in {"3"}:
            return attn_0_7_output == "3"
        elif attn_0_0_output in {"<s>"}:
            return attn_0_7_output == "<pad>"

    num_attn_2_5_pattern = select(attn_0_7_outputs, attn_0_0_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_4_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_1_output, attn_0_4_output):
        if attn_0_1_output in {"<s>", "1", "0", "5", "3", "4"}:
            return attn_0_4_output == ""
        elif attn_0_1_output in {"2"}:
            return attn_0_4_output == "2"

    num_attn_2_6_pattern = select(attn_0_4_outputs, attn_0_1_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_3_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_0_output, attn_0_1_output):
        if attn_0_0_output in {"<s>", "0", "5", "2", "4"}:
            return attn_0_1_output == ""
        elif attn_0_0_output in {"3", "1"}:
            return attn_0_1_output == "1"

    num_attn_2_7_pattern = select(attn_0_1_outputs, attn_0_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_1_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(position, num_mlp_0_3_output):
        key = (position, num_mlp_0_3_output)
        return 27

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(positions, num_mlp_0_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_2_output, attn_0_4_output):
        key = (attn_1_2_output, attn_0_4_output)
        return 22

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_0_4_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(mlp_1_0_output, num_mlp_1_0_output):
        key = (mlp_1_0_output, num_mlp_1_0_output)
        return 24

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(mlp_1_0_outputs, num_mlp_1_0_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(token, attn_0_1_output):
        key = (token, attn_0_1_output)
        return 9

    mlp_2_3_outputs = [mlp_2_3(k0, k1) for k0, k1 in zip(tokens, attn_0_1_outputs)]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_7_output, num_attn_2_0_output):
        key = (num_attn_2_7_output, num_attn_2_0_output)
        return 34

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_7_output, num_attn_1_4_output):
        key = (num_attn_2_7_output, num_attn_1_4_output)
        return 6

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_3_output, num_attn_2_6_output):
        key = (num_attn_1_3_output, num_attn_2_6_output)
        return 12

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_2_6_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_6_output, num_attn_1_6_output):
        key = (num_attn_2_6_output, num_attn_1_6_output)
        return 4

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_6_outputs, num_attn_1_6_outputs)
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


print(run(["<s>", "3", "4", "0", "1", "3", "5"]))
