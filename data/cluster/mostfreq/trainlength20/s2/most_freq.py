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
        "output/length/rasp/mostfreq/trainlength20/s2/most_freq_weights.csv",
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
        elif position in {1, 9}:
            return token == "0"
        elif position in {2, 7, 10, 11, 12, 13, 14, 16, 17, 19, 29}:
            return token == "1"
        elif position in {3, 4, 5, 6}:
            return token == "5"
        elif position in {8}:
            return token == "4"
        elif position in {15, 18, 20, 21, 23, 24, 25, 26, 27, 28}:
            return token == ""
        elif position in {22}:
            return token == "<pad>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 7, 9, 11, 13, 14, 16, 18, 19}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 12
        elif q_position in {3, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 15
        elif q_position in {25, 6}:
            return k_position == 17
        elif q_position in {8, 17, 12, 15}:
            return k_position == 4
        elif q_position in {10}:
            return k_position == 3
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {22, 23}:
            return k_position == 29
        elif q_position in {24}:
            return k_position == 13
        elif q_position in {26}:
            return k_position == 20
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 8
        elif q_position in {29}:
            return k_position == 25

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 1
        elif q_position in {1, 28}:
            return k_position == 20
        elif q_position in {2}:
            return k_position == 0
        elif q_position in {4}:
            return k_position == 8
        elif q_position in {5, 8, 10, 12, 13, 18}:
            return k_position == 2
        elif q_position in {9, 19, 6, 15}:
            return k_position == 3
        elif q_position in {17, 7}:
            return k_position == 4
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {16, 14, 23}:
            return k_position == 6
        elif q_position in {24, 20}:
            return k_position == 23
        elif q_position in {21}:
            return k_position == 29
        elif q_position in {25, 27, 22}:
            return k_position == 18
        elif q_position in {26, 29}:
            return k_position == 28

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 4, 5, 6, 11}:
            return token == "4"
        elif position in {1}:
            return token == "0"
        elif position in {19, 2, 18}:
            return token == "5"
        elif position in {3, 7, 8, 9, 10, 12, 13}:
            return token == "2"
        elif position in {14, 20, 21, 23, 24, 25, 27, 28, 29}:
            return token == ""
        elif position in {15}:
            return token == "1"
        elif position in {16, 17}:
            return token == "3"
        elif position in {26, 22}:
            return token == "<pad>"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 11, 14, 15}:
            return token == "2"
        elif position in {1, 2, 3, 4}:
            return token == "0"
        elif position in {5, 6}:
            return token == "5"
        elif position in {7, 8, 9, 12, 16, 19}:
            return token == "3"
        elif position in {10, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {13}:
            return token == "4"

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 2
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {8, 4, 15}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {10, 6}:
            return k_position == 5
        elif q_position in {19, 7}:
            return k_position == 4
        elif q_position in {9, 18}:
            return k_position == 6
        elif q_position in {25, 29, 11, 21}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {28, 13}:
            return k_position == 9
        elif q_position in {14}:
            return k_position == 11
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17, 20}:
            return k_position == 18
        elif q_position in {22}:
            return k_position == 16
        elif q_position in {24, 27, 23}:
            return k_position == 8
        elif q_position in {26}:
            return k_position == 21

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 3, 15}:
            return k_position == 3
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {18, 4}:
            return k_position == 5
        elif q_position in {19, 12, 5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8, 23}:
            return k_position == 12
        elif q_position in {9, 27}:
            return k_position == 4
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {25, 13}:
            return k_position == 16
        elif q_position in {20, 29, 14, 22}:
            return k_position == 9
        elif q_position in {16}:
            return k_position == 8
        elif q_position in {17}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {24}:
            return k_position == 11
        elif q_position in {26}:
            return k_position == 24
        elif q_position in {28}:
            return k_position == 22

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1, 2, 3, 4, 5, 6, 9, 10, 11, 15, 16}:
            return token == "1"
        elif position in {14, 7}:
            return token == "5"
        elif position in {8, 27}:
            return token == "3"
        elif position in {12, 13, 18, 20, 21, 22, 23, 25, 28, 29}:
            return token == ""
        elif position in {17}:
            return token == "4"
        elif position in {19}:
            return token == "0"
        elif position in {24}:
            return token == "<s>"
        elif position in {26}:
            return token == "<pad>"

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 5, 6, 14, 16}:
            return token == "<s>"
        elif position in {1, 7, 8, 9, 11}:
            return token == "4"
        elif position in {
            2,
            3,
            4,
            10,
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
        }:
            return token == ""

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
            return token == ""
        elif position in {4, 5, 6}:
            return token == "0"
        elif position in {7}:
            return token == "<s>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 1, 2}:
            return token == "2"
        elif position in {3, 4}:
            return token == "<s>"
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
        }:
            return token == ""

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1, 7}:
            return token == "5"
        elif position in {
            2,
            3,
            4,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
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
        }:
            return token == ""
        elif position in {5, 6}:
            return token == "<s>"
        elif position in {16, 18}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 8, 9, 7}:
            return token == "1"
        elif position in {1}:
            return token == "0"
        elif position in {
            2,
            3,
            4,
            5,
            6,
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
            26,
            27,
            28,
            29,
        }:
            return token == ""
        elif position in {14}:
            return token == "<pad>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 1, 7}:
            return token == "3"
        elif position in {2, 5, 6}:
            return token == "<s>"
        elif position in {
            3,
            4,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
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
            return token == ""
        elif position in {16, 17}:
            return token == "<pad>"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 5, 6, 22, 24}:
            return token == "<pad>"
        elif position in {1, 11, 13, 17}:
            return token == "2"
        elif position in {2, 3, 4, 20, 21, 23, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {12, 7}:
            return token == "3"
        elif position in {8, 9, 10, 15, 16, 19}:
            return token == "1"
        elif position in {14}:
            return token == "4"
        elif position in {18}:
            return token == "5"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 8, 7}:
            return token == "<s>"
        elif position in {1}:
            return token == "2"
        elif position in {
            2,
            3,
            4,
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
            return token == ""
        elif position in {5, 6}:
            return token == "3"
        elif position in {9, 10, 11, 12}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output):
        key = attn_0_2_output
        return 1

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in attn_0_2_outputs]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_6_output, attn_0_3_output):
        key = (attn_0_6_output, attn_0_3_output)
        if key in {
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "<s>"),
            ("2", "<s>"),
            ("3", "<s>"),
            ("5", "<s>"),
            ("<s>", "<s>"),
        }:
            return 12
        return 5

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_3_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(position, attn_0_6_output):
        key = (position, attn_0_6_output)
        if key in {
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "<s>"),
            (11, "0"),
            (11, "1"),
            (11, "2"),
            (11, "3"),
            (11, "4"),
            (11, "5"),
            (11, "<s>"),
            (21, "2"),
            (23, "2"),
            (25, "2"),
            (26, "2"),
            (27, "2"),
        }:
            return 19
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
            (8, "0"),
            (8, "1"),
            (8, "2"),
            (8, "3"),
            (8, "4"),
            (8, "5"),
            (8, "<s>"),
        }:
            return 8
        elif key in {
            (1, "0"),
            (1, "1"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "<s>"),
        }:
            return 28
        elif key in {
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "<s>"),
        }:
            return 25
        elif key in {
            (1, "4"),
            (1, "<s>"),
            (20, "2"),
            (20, "4"),
            (21, "4"),
            (22, "2"),
            (22, "4"),
            (23, "4"),
            (24, "2"),
            (24, "4"),
            (25, "4"),
            (26, "4"),
            (27, "4"),
            (28, "2"),
            (28, "4"),
            (29, "2"),
            (29, "4"),
        }:
            return 17
        elif key in {(1, "2"), (1, "3"), (1, "5")}:
            return 3
        return 26

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(positions, attn_0_6_outputs)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position, attn_0_4_output):
        key = (position, attn_0_4_output)
        if key in {
            (0, "3"),
            (1, "0"),
            (1, "1"),
            (1, "4"),
            (1, "5"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
            (7, "1"),
            (7, "4"),
            (7, "5"),
            (7, "<s>"),
            (9, "1"),
            (9, "5"),
            (9, "<s>"),
            (10, "0"),
            (10, "1"),
            (10, "4"),
            (10, "5"),
            (10, "<s>"),
            (12, "1"),
            (12, "4"),
            (12, "5"),
            (13, "0"),
            (13, "1"),
            (13, "4"),
            (13, "5"),
            (13, "<s>"),
            (14, "0"),
            (14, "1"),
            (14, "4"),
            (14, "5"),
            (14, "<s>"),
            (15, "1"),
            (15, "4"),
            (15, "5"),
            (15, "<s>"),
            (16, "1"),
            (16, "<s>"),
            (18, "0"),
            (18, "1"),
            (18, "4"),
            (18, "5"),
            (18, "<s>"),
            (19, "5"),
            (19, "<s>"),
            (20, "1"),
            (20, "4"),
            (20, "5"),
            (20, "<s>"),
            (21, "1"),
            (21, "4"),
            (21, "5"),
            (22, "1"),
            (22, "4"),
            (22, "5"),
            (23, "1"),
            (23, "4"),
            (23, "5"),
            (23, "<s>"),
            (24, "1"),
            (24, "4"),
            (24, "5"),
            (25, "1"),
            (25, "4"),
            (25, "5"),
            (25, "<s>"),
            (26, "1"),
            (26, "4"),
            (26, "5"),
            (26, "<s>"),
            (27, "0"),
            (27, "1"),
            (27, "4"),
            (27, "5"),
            (28, "0"),
            (28, "1"),
            (28, "4"),
            (28, "5"),
            (29, "1"),
            (29, "4"),
            (29, "5"),
            (29, "<s>"),
        }:
            return 7
        elif key in {
            (0, "1"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (21, "<s>"),
            (27, "<s>"),
            (28, "<s>"),
        }:
            return 16
        elif key in {
            (5, "1"),
            (5, "4"),
            (5, "5"),
            (5, "<s>"),
            (6, "<s>"),
            (22, "<s>"),
            (24, "<s>"),
        }:
            return 3
        elif key in {(1, "<s>")}:
            return 0
        elif key in {(0, "0"), (0, "2")}:
            return 9
        elif key in {(3, "<s>"), (12, "<s>")}:
            return 15
        elif key in {(17, "1"), (19, "1")}:
            return 27
        return 5

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(positions, attn_0_4_outputs)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_7_output, num_attn_0_6_output):
        key = (num_attn_0_7_output, num_attn_0_6_output)
        return 3

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_4_output):
        key = num_attn_0_4_output
        return 3

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_4_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_0_output, num_attn_0_6_output):
        key = (num_attn_0_0_output, num_attn_0_6_output)
        return 4

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_1_output, num_attn_0_5_output):
        key = (num_attn_0_1_output, num_attn_0_5_output)
        return 2

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_2_output, token):
        if mlp_0_2_output in {0, 4, 7, 10, 11, 12, 13, 16, 18, 20, 24, 29}:
            return token == ""
        elif mlp_0_2_output in {1, 8, 14, 17, 21, 22, 23, 25, 27, 28}:
            return token == "3"
        elif mlp_0_2_output in {2}:
            return token == "<pad>"
        elif mlp_0_2_output in {3}:
            return token == "5"
        elif mlp_0_2_output in {5}:
            return token == "4"
        elif mlp_0_2_output in {26, 6}:
            return token == "<s>"
        elif mlp_0_2_output in {9, 19}:
            return token == "2"
        elif mlp_0_2_output in {15}:
            return token == "1"

    attn_1_0_pattern = select_closest(tokens, mlp_0_2_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"<s>", "0"}:
            return k_token == "<s>"
        elif q_token in {"5", "3", "1", "4"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "3"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_0_output, mlp_0_2_output):
        if attn_0_0_output in {"0"}:
            return mlp_0_2_output == 6
        elif attn_0_0_output in {"3", "1", "<s>", "4", "5"}:
            return mlp_0_2_output == 9
        elif attn_0_0_output in {"2"}:
            return mlp_0_2_output == 27

    attn_1_2_pattern = select_closest(mlp_0_2_outputs, attn_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, mlp_0_2_output):
        if token in {"<s>", "0"}:
            return mlp_0_2_output == 22
        elif token in {"1"}:
            return mlp_0_2_output == 28
        elif token in {"2"}:
            return mlp_0_2_output == 8
        elif token in {"3"}:
            return mlp_0_2_output == 3
        elif token in {"4"}:
            return mlp_0_2_output == 0
        elif token in {"5"}:
            return mlp_0_2_output == 19

    attn_1_3_pattern = select_closest(mlp_0_2_outputs, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_6_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_token, k_token):
        if q_token in {"2", "1", "0"}:
            return k_token == "4"
        elif q_token in {"3", "<s>"}:
            return k_token == ""
        elif q_token in {"4"}:
            return k_token == "<s>"
        elif q_token in {"5"}:
            return k_token == "3"

    attn_1_4_pattern = select_closest(tokens, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, mlp_0_3_output):
        if position in {0, 22, 25, 26, 28}:
            return mlp_0_3_output == 24
        elif position in {1, 2, 3}:
            return mlp_0_3_output == 5
        elif position in {4, 5, 6, 7, 8, 19, 20, 23}:
            return mlp_0_3_output == 0
        elif position in {9, 11, 13, 14, 15, 17}:
            return mlp_0_3_output == 18
        elif position in {10, 18, 12}:
            return mlp_0_3_output == 12
        elif position in {16}:
            return mlp_0_3_output == 9
        elif position in {21}:
            return mlp_0_3_output == 7
        elif position in {24}:
            return mlp_0_3_output == 14
        elif position in {27, 29}:
            return mlp_0_3_output == 22

    attn_1_5_pattern = select_closest(mlp_0_3_outputs, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, mlp_0_2_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(mlp_0_2_output, position):
        if mlp_0_2_output in {0, 26, 20}:
            return position == 9
        elif mlp_0_2_output in {1, 29}:
            return position == 16
        elif mlp_0_2_output in {8, 2, 3}:
            return position == 1
        elif mlp_0_2_output in {4}:
            return position == 6
        elif mlp_0_2_output in {5, 6}:
            return position == 11
        elif mlp_0_2_output in {7, 10, 11, 12, 13, 18, 23}:
            return position == 0
        elif mlp_0_2_output in {16, 9}:
            return position == 7
        elif mlp_0_2_output in {14}:
            return position == 8
        elif mlp_0_2_output in {15}:
            return position == 15
        elif mlp_0_2_output in {24, 17, 25}:
            return position == 12
        elif mlp_0_2_output in {19}:
            return position == 3
        elif mlp_0_2_output in {27, 28, 21}:
            return position == 4
        elif mlp_0_2_output in {22}:
            return position == 2

    attn_1_6_pattern = select_closest(positions, mlp_0_2_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_4_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(token, mlp_0_3_output):
        if token in {"0"}:
            return mlp_0_3_output == 7
        elif token in {"5", "1", "4"}:
            return mlp_0_3_output == 27
        elif token in {"2"}:
            return mlp_0_3_output == 15
        elif token in {"3"}:
            return mlp_0_3_output == 5
        elif token in {"<s>"}:
            return mlp_0_3_output == 16

    attn_1_7_pattern = select_closest(mlp_0_3_outputs, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_0_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_2_output, attn_0_3_output):
        if mlp_0_2_output in {0, 4, 5, 6, 9, 10, 14, 19, 20, 21, 23, 27, 29}:
            return attn_0_3_output == ""
        elif mlp_0_2_output in {1, 2, 28, 22}:
            return attn_0_3_output == "4"
        elif mlp_0_2_output in {17, 3}:
            return attn_0_3_output == "1"
        elif mlp_0_2_output in {7, 8, 11, 12, 13, 15, 16, 18, 25}:
            return attn_0_3_output == "<pad>"
        elif mlp_0_2_output in {24, 26}:
            return attn_0_3_output == "0"

    num_attn_1_0_pattern = select(attn_0_3_outputs, mlp_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_4_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_2_output, attn_0_0_output):
        if mlp_0_2_output in {
            0,
            1,
            3,
            4,
            5,
            6,
            7,
            10,
            11,
            12,
            14,
            15,
            17,
            18,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
        }:
            return attn_0_0_output == ""
        elif mlp_0_2_output in {2, 8, 9, 13, 16, 19, 20, 21}:
            return attn_0_0_output == "1"

    num_attn_1_1_pattern = select(attn_0_0_outputs, mlp_0_2_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_4_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_3_output, attn_0_0_output):
        if attn_0_3_output in {"3", "1", "0"}:
            return attn_0_0_output == ""
        elif attn_0_3_output in {"2", "<s>"}:
            return attn_0_0_output == "2"
        elif attn_0_3_output in {"5", "4"}:
            return attn_0_0_output == "<pad>"

    num_attn_1_2_pattern = select(attn_0_0_outputs, attn_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(mlp_0_2_output, attn_0_5_output):
        if mlp_0_2_output in {0, 1, 2, 3, 12, 14, 16, 17, 18, 21, 22, 23, 25, 27, 28}:
            return attn_0_5_output == ""
        elif mlp_0_2_output in {4, 5, 6, 7, 8, 9, 19, 20, 24, 26, 29}:
            return attn_0_5_output == "4"
        elif mlp_0_2_output in {10}:
            return attn_0_5_output == "3"
        elif mlp_0_2_output in {11}:
            return attn_0_5_output == "5"
        elif mlp_0_2_output in {13, 15}:
            return attn_0_5_output == "<pad>"

    num_attn_1_3_pattern = select(attn_0_5_outputs, mlp_0_2_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(token, position):
        if token in {"0"}:
            return position == 15
        elif token in {"1"}:
            return position == 21
        elif token in {"2"}:
            return position == 22
        elif token in {"3"}:
            return position == 23
        elif token in {"4"}:
            return position == 5
        elif token in {"5"}:
            return position == 27
        elif token in {"<s>"}:
            return position == 19

    num_attn_1_4_pattern = select(positions, tokens, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_0_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_3_output, attn_0_5_output):
        if attn_0_3_output in {"2", "3", "1", "0"}:
            return attn_0_5_output == "5"
        elif attn_0_3_output in {"5", "4"}:
            return attn_0_5_output == ""
        elif attn_0_3_output in {"<s>"}:
            return attn_0_5_output == "<pad>"

    num_attn_1_5_pattern = select(attn_0_5_outputs, attn_0_3_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_5_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(token, position):
        if token in {"0"}:
            return position == 18
        elif token in {"1"}:
            return position == 22
        elif token in {"2"}:
            return position == 6
        elif token in {"3"}:
            return position == 19
        elif token in {"4"}:
            return position == 5
        elif token in {"5"}:
            return position == 26
        elif token in {"<s>"}:
            return position == 7

    num_attn_1_6_pattern = select(positions, tokens, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_5_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, attn_0_5_output):
        if position in {0, 1}:
            return attn_0_5_output == "5"
        elif position in {2}:
            return attn_0_5_output == "2"
        elif position in {3, 4}:
            return attn_0_5_output == "<s>"
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
            15,
            16,
            19,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            29,
        }:
            return attn_0_5_output == ""
        elif position in {8, 17, 18}:
            return attn_0_5_output == "<pad>"
        elif position in {25, 20}:
            return attn_0_5_output == "4"

    num_attn_1_7_pattern = select(attn_0_5_outputs, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_3_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output, attn_0_0_output):
        key = (attn_1_0_output, attn_0_0_output)
        if key in {
            ("0", "2"),
            ("0", "3"),
            ("0", "<s>"),
            ("1", "2"),
            ("1", "3"),
            ("1", "<s>"),
            ("2", "2"),
            ("2", "3"),
            ("2", "<s>"),
            ("3", "2"),
            ("3", "3"),
            ("3", "<s>"),
            ("4", "2"),
            ("4", "3"),
            ("4", "<s>"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "<s>"),
        }:
            return 25
        return 3

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_0_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_5_output, mlp_0_3_output):
        key = (attn_0_5_output, mlp_0_3_output)
        if key in {
            ("0", 15),
            ("0", 27),
            ("1", 15),
            ("1", 27),
            ("2", 15),
            ("2", 27),
            ("3", 15),
            ("3", 27),
            ("4", 15),
            ("4", 27),
            ("5", 15),
            ("5", 27),
            ("<s>", 15),
            ("<s>", 27),
        }:
            return 9
        return 22

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_5_outputs, mlp_0_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(num_mlp_0_2_output, attn_0_0_output):
        key = (num_mlp_0_2_output, attn_0_0_output)
        return 12

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(num_mlp_0_2_outputs, attn_0_0_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_5_output, attn_1_2_output):
        key = (attn_1_5_output, attn_1_2_output)
        if key in {(22, 22)}:
            return 26
        return 20

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_5_outputs, attn_1_2_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_5_output):
        key = num_attn_0_5_output
        if key in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}:
            return 19
        return 29

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_5_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_1_1_output):
        key = (num_attn_1_2_output, num_attn_1_1_output)
        return 7

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_4_output, num_attn_1_3_output):
        key = (num_attn_1_4_output, num_attn_1_3_output)
        return 0

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_0_output, num_attn_1_6_output):
        key = (num_attn_0_0_output, num_attn_1_6_output)
        return 5

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, attn_0_0_output):
        if token in {"3", "1", "4", "0", "2"}:
            return attn_0_0_output == "5"
        elif token in {"5"}:
            return attn_0_0_output == "0"
        elif token in {"<s>"}:
            return attn_0_0_output == "<pad>"

    attn_2_0_pattern = select_closest(attn_0_0_outputs, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_1_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, mlp_0_2_output):
        if position in {0, 4, 14, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29}:
            return mlp_0_2_output == 22
        elif position in {1, 2, 3}:
            return mlp_0_2_output == 26
        elif position in {5, 6}:
            return mlp_0_2_output == 25
        elif position in {7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18}:
            return mlp_0_2_output == 19
        elif position in {21}:
            return mlp_0_2_output == 24

    attn_2_1_pattern = select_closest(mlp_0_2_outputs, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, mlp_0_3_output):
        if position in {0, 7, 21, 22, 23, 24, 25, 26, 28}:
            return mlp_0_3_output == 24
        elif position in {19, 1, 11, 13}:
            return mlp_0_3_output == 28
        elif position in {2, 8, 18, 20, 27, 29}:
            return mlp_0_3_output == 0
        elif position in {3}:
            return mlp_0_3_output == 27
        elif position in {4}:
            return mlp_0_3_output == 22
        elif position in {5}:
            return mlp_0_3_output == 15
        elif position in {6}:
            return mlp_0_3_output == 26
        elif position in {9}:
            return mlp_0_3_output == 18
        elif position in {10, 12, 14, 15, 16}:
            return mlp_0_3_output == 3
        elif position in {17}:
            return mlp_0_3_output == 5

    attn_2_2_pattern = select_closest(mlp_0_3_outputs, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 0
        elif q_position in {1, 22}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {5, 6}:
            return k_position == 8
        elif q_position in {7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 25, 29}:
            return k_position == 6
        elif q_position in {13}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 17
        elif q_position in {21}:
            return k_position == 27
        elif q_position in {23}:
            return k_position == 25
        elif q_position in {24}:
            return k_position == 16
        elif q_position in {26}:
            return k_position == 28
        elif q_position in {27}:
            return k_position == 4
        elif q_position in {28}:
            return k_position == 7

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_0_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_0_output, position):
        if attn_0_0_output in {"0"}:
            return position == 9
        elif attn_0_0_output in {"5", "1", "<s>", "4"}:
            return position == 0
        elif attn_0_0_output in {"2"}:
            return position == 22
        elif attn_0_0_output in {"3"}:
            return position == 3

    attn_2_4_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, mlp_0_3_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(token, position):
        if token in {"4", "0"}:
            return position == 6
        elif token in {"1"}:
            return position == 1
        elif token in {"2"}:
            return position == 22
        elif token in {"3"}:
            return position == 12
        elif token in {"5"}:
            return position == 13
        elif token in {"<s>"}:
            return position == 24

    attn_2_5_pattern = select_closest(positions, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_3_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(position, mlp_0_2_output):
        if position in {0, 25, 26, 27, 28}:
            return mlp_0_2_output == 27
        elif position in {1, 7}:
            return mlp_0_2_output == 28
        elif position in {2, 11, 14}:
            return mlp_0_2_output == 22
        elif position in {3, 4, 5, 6}:
            return mlp_0_2_output == 19
        elif position in {8, 9, 10, 12, 15, 16, 17, 18}:
            return mlp_0_2_output == 3
        elif position in {13}:
            return mlp_0_2_output == 14
        elif position in {19}:
            return mlp_0_2_output == 24
        elif position in {20}:
            return mlp_0_2_output == 9
        elif position in {24, 21, 22, 23}:
            return mlp_0_2_output == 23
        elif position in {29}:
            return mlp_0_2_output == 5

    attn_2_6_pattern = select_closest(mlp_0_2_outputs, positions, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, mlp_0_1_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(position, mlp_0_2_output):
        if position in {0, 20, 21, 22, 25, 27, 29}:
            return mlp_0_2_output == 20
        elif position in {1, 3, 4, 5, 6}:
            return mlp_0_2_output == 19
        elif position in {2}:
            return mlp_0_2_output == 27
        elif position in {15, 7}:
            return mlp_0_2_output == 3
        elif position in {8, 9, 10, 11, 12, 13, 16}:
            return mlp_0_2_output == 28
        elif position in {14}:
            return mlp_0_2_output == 17
        elif position in {17}:
            return mlp_0_2_output == 26
        elif position in {18}:
            return mlp_0_2_output == 29
        elif position in {19}:
            return mlp_0_2_output == 13
        elif position in {24, 26, 28, 23}:
            return mlp_0_2_output == 9

    attn_2_7_pattern = select_closest(mlp_0_2_outputs, positions, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, mlp_0_1_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_2_output, attn_0_5_output):
        if mlp_0_2_output in {0, 8, 9, 12, 19, 20}:
            return attn_0_5_output == "2"
        elif mlp_0_2_output in {
            1,
            2,
            3,
            4,
            5,
            6,
            10,
            13,
            14,
            15,
            16,
            17,
            18,
            21,
            22,
            23,
            24,
            25,
            27,
            28,
            29,
        }:
            return attn_0_5_output == ""
        elif mlp_0_2_output in {11, 7}:
            return attn_0_5_output == "<s>"
        elif mlp_0_2_output in {26}:
            return attn_0_5_output == "5"

    num_attn_2_0_pattern = select(attn_0_5_outputs, mlp_0_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_0_2_output, attn_1_7_output):
        if mlp_0_2_output in {0, 8, 21, 24, 25, 29}:
            return attn_1_7_output == "<s>"
        elif mlp_0_2_output in {1, 2, 3, 5, 7, 10, 11, 12, 13, 14, 16, 17, 18, 22, 28}:
            return attn_1_7_output == ""
        elif mlp_0_2_output in {27, 4}:
            return attn_1_7_output == "<pad>"
        elif mlp_0_2_output in {6, 9, 15, 19, 20, 23}:
            return attn_1_7_output == "5"
        elif mlp_0_2_output in {26}:
            return attn_1_7_output == "2"

    num_attn_2_1_pattern = select(attn_1_7_outputs, mlp_0_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_7_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_0_output, attn_0_5_output):
        if attn_0_0_output in {"3", "1", "<s>", "4", "0", "2"}:
            return attn_0_5_output == ""
        elif attn_0_0_output in {"5"}:
            return attn_0_5_output == "5"

    num_attn_2_2_pattern = select(attn_0_5_outputs, attn_0_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_6_output, attn_0_4_output):
        if attn_0_6_output in {"0"}:
            return attn_0_4_output == "0"
        elif attn_0_6_output in {"3", "1", "<s>", "4", "2", "5"}:
            return attn_0_4_output == ""

    num_attn_2_3_pattern = select(attn_0_4_outputs, attn_0_6_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_4_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(token, position):
        if token in {"1", "0"}:
            return position == 25
        elif token in {"2"}:
            return position == 19
        elif token in {"3"}:
            return position == 21
        elif token in {"4"}:
            return position == 28
        elif token in {"5"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 15

    num_attn_2_4_pattern = select(positions, tokens, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_5_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(mlp_0_2_output, attn_0_7_output):
        if mlp_0_2_output in {
            0,
            1,
            2,
            3,
            5,
            7,
            8,
            12,
            13,
            14,
            15,
            17,
            18,
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
            return attn_0_7_output == ""
        elif mlp_0_2_output in {16, 10, 4}:
            return attn_0_7_output == "<pad>"
        elif mlp_0_2_output in {6}:
            return attn_0_7_output == "1"
        elif mlp_0_2_output in {9, 19, 20}:
            return attn_0_7_output == "5"
        elif mlp_0_2_output in {11}:
            return attn_0_7_output == "0"

    num_attn_2_5_pattern = select(attn_0_7_outputs, mlp_0_2_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_3_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(position, token):
        if position in {0, 26}:
            return token == "3"
        elif position in {1}:
            return token == "1"
        elif position in {
            2,
            4,
            5,
            6,
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
            23,
            25,
            27,
            28,
        }:
            return token == ""
        elif position in {8, 3, 7}:
            return token == "<pad>"
        elif position in {24, 29, 22}:
            return token == "<s>"

    num_attn_2_6_pattern = select(tokens, positions, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, ones)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(mlp_0_2_output, attn_0_5_output):
        if mlp_0_2_output in {0, 1, 2, 3, 5, 6, 7, 12, 17, 18, 22, 25, 27, 28}:
            return attn_0_5_output == ""
        elif mlp_0_2_output in {4, 8, 9, 13, 14, 15, 16, 19, 20, 21, 23, 24, 26, 29}:
            return attn_0_5_output == "3"
        elif mlp_0_2_output in {10}:
            return attn_0_5_output == "5"
        elif mlp_0_2_output in {11}:
            return attn_0_5_output == "0"

    num_attn_2_7_pattern = select(attn_0_5_outputs, mlp_0_2_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_5_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_3_output, num_mlp_1_1_output):
        key = (attn_0_3_output, num_mlp_1_1_output)
        return 2

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_7_output, attn_0_7_output):
        key = (attn_1_7_output, attn_0_7_output)
        if key in {
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("1", "<s>"),
            ("2", "1"),
            ("3", "1"),
            ("4", "1"),
            ("5", "1"),
            ("<s>", "1"),
        }:
            return 16
        return 8

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_7_outputs, attn_0_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(token, position):
        key = (token, position)
        if key in {("0", 1), ("4", 1)}:
            return 26
        return 13

    mlp_2_2_outputs = [mlp_2_2(k0, k1) for k0, k1 in zip(tokens, positions)]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_1_7_output, mlp_1_1_output):
        key = (attn_1_7_output, mlp_1_1_output)
        return 23

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_1_7_outputs, mlp_1_1_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_2_7_output):
        key = (num_attn_1_2_output, num_attn_2_7_output)
        return 1

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_7_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_6_output, num_attn_1_3_output):
        key = (num_attn_1_6_output, num_attn_1_3_output)
        return 6

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_7_output, num_attn_2_4_output):
        key = (num_attn_1_7_output, num_attn_2_4_output)
        return 26

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_2_4_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_2_output, num_attn_2_2_output):
        key = (num_attn_1_2_output, num_attn_2_2_output)
        return 29

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_2_outputs)
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


print(run(["<s>", "5", "0", "3", "2", "3", "0", "2", "1", "3"]))
