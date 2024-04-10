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
        "output/length/rasp/mostfreq/trainlength20/s3/most_freq_weights.csv",
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
        if position in {0, 3, 4, 5, 6, 8, 9, 10, 12, 13}:
            return token == "1"
        elif position in {1, 28, 22}:
            return token == "5"
        elif position in {2, 20}:
            return token == "2"
        elif position in {7}:
            return token == "4"
        elif position in {11, 14, 16, 19, 21, 23, 24, 25, 26, 27, 29}:
            return token == ""
        elif position in {18, 15}:
            return token == "<s>"
        elif position in {17}:
            return token == "3"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {8, 1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {9, 27, 4, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {25, 7}:
            return k_position == 5
        elif q_position in {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return k_position == 0
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {21, 22}:
            return k_position == 15
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 29
        elif q_position in {28}:
            return k_position == 19
        elif q_position in {29}:
            return k_position == 24

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 11, 13, 7}:
            return token == "1"
        elif position in {16, 1, 18, 14}:
            return token == "2"
        elif position in {2, 3, 4, 5, 6}:
            return token == "0"
        elif position in {8, 17, 10, 15}:
            return token == "4"
        elif position in {9}:
            return token == "3"
        elif position in {12, 20, 21, 23, 25, 26, 27, 28}:
            return token == ""
        elif position in {19}:
            return token == "<s>"
        elif position in {24, 29, 22}:
            return token == "5"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 8}:
            return token == "4"
        elif position in {24, 1, 27, 28}:
            return token == "5"
        elif position in {2, 9, 10, 12, 15, 21}:
            return token == "1"
        elif position in {3, 4, 5, 6}:
            return token == "2"
        elif position in {7}:
            return token == "<pad>"
        elif position in {11, 13, 14, 16, 17, 18, 19}:
            return token == "<s>"
        elif position in {20, 22, 23, 25, 26, 29}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 8, 10, 13}:
            return token == "1"
        elif position in {1, 2}:
            return token == "2"
        elif position in {3, 4, 5, 6}:
            return token == "3"
        elif position in {11, 7}:
            return token == "4"
        elif position in {9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {12, 14, 15, 16, 17, 18, 19}:
            return token == "<s>"

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(token, position):
        if token in {"0", "4"}:
            return position == 6
        elif token in {"1"}:
            return position == 2
        elif token in {"2"}:
            return position == 4
        elif token in {"5", "<s>", "3"}:
            return position == 5

    attn_0_5_pattern = select_closest(positions, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, positions)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 21}:
            return k_position == 9
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {18, 3}:
            return k_position == 3
        elif q_position in {25, 4, 23}:
            return k_position == 5
        elif q_position in {5, 14}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 12
        elif q_position in {7}:
            return k_position == 19
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {10, 22}:
            return k_position == 4
        elif q_position in {11, 24, 26, 27, 28}:
            return k_position == 8
        elif q_position in {12, 15}:
            return k_position == 10
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {16, 19}:
            return k_position == 0
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {20}:
            return k_position == 14
        elif q_position in {29}:
            return k_position == 21

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0, 9, 15, 7}:
            return token == "4"
        elif position in {1}:
            return token == "3"
        elif position in {2, 3, 4, 5, 6}:
            return token == "5"
        elif position in {8, 11, 12, 13, 14, 16}:
            return token == "1"
        elif position in {10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {17, 19}:
            return token == "2"
        elif position in {18}:
            return token == "<pad>"

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
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
        }:
            return token == ""
        elif position in {1}:
            return token == "5"
        elif position in {5, 6}:
            return token == "2"
        elif position in {9, 7}:
            return token == "<s>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 1, 2}:
            return token == "0"
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

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 1, 2, 3, 7}:
            return token == "2"
        elif position in {4, 5}:
            return token == "<s>"
        elif position in {
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
        }:
            return token == ""

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1}:
            return token == "2"
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
            16,
            17,
            18,
            19,
            21,
            23,
            24,
            26,
            27,
            28,
            29,
        }:
            return token == ""
        elif position in {5, 6}:
            return token == "0"
        elif position in {7}:
            return token == "<s>"
        elif position in {25, 20, 22}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 1, 2, 3, 4}:
            return token == "3"
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

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 9, 12, 13, 14, 16, 19}:
            return token == "<s>"
        elif position in {1, 7, 8, 11, 17, 18}:
            return token == "1"
        elif position in {2, 3, 4, 5, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {6}:
            return token == "<pad>"
        elif position in {10, 15}:
            return token == "2"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 5, 6}:
            return token == "4"
        elif position in {1}:
            return token == "1"
        elif position in {
            2,
            3,
            7,
            8,
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
        }:
            return token == ""
        elif position in {4}:
            return token == "<s>"
        elif position in {12}:
            return token == "<pad>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 2, 3, 4, 5, 6, 20, 21, 24, 25, 26, 27, 29}:
            return token == ""
        elif position in {1}:
            return token == "0"
        elif position in {18, 7}:
            return token == "3"
        elif position in {8, 11, 13, 15, 17}:
            return token == "1"
        elif position in {9, 19}:
            return token == "<s>"
        elif position in {16, 10}:
            return token == "2"
        elif position in {12, 14}:
            return token == "4"
        elif position in {28, 22, 23}:
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
            ("1", 2),
            ("1", 20),
            ("1", 21),
            ("1", 22),
            ("1", 23),
            ("1", 24),
            ("1", 25),
            ("1", 26),
            ("1", 27),
            ("1", 28),
            ("1", 29),
            ("2", 2),
            ("3", 2),
            ("4", 2),
            ("4", 20),
            ("4", 21),
            ("4", 22),
            ("4", 23),
            ("4", 24),
            ("4", 25),
            ("4", 26),
            ("4", 27),
            ("4", 28),
            ("4", 29),
            ("5", 2),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 20
        elif key in {
            ("0", 0),
            ("0", 3),
            ("1", 3),
            ("2", 0),
            ("2", 1),
            ("2", 3),
            ("3", 0),
            ("3", 1),
            ("3", 3),
            ("4", 0),
            ("4", 3),
            ("4", 10),
            ("5", 0),
            ("5", 3),
        }:
            return 6
        elif key in {
            ("0", 10),
            ("0", 18),
            ("1", 10),
            ("1", 18),
            ("2", 10),
            ("2", 18),
            ("3", 10),
            ("3", 18),
            ("4", 18),
            ("5", 10),
            ("5", 18),
            ("<s>", 10),
            ("<s>", 18),
        }:
            return 7
        elif key in {("0", 1), ("1", 0), ("1", 1), ("5", 1)}:
            return 1
        elif key in {("4", 1)}:
            return 13
        return 11

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_7_output):
        key = (position, attn_0_7_output)
        if key in {
            (0, "1"),
            (1, "5"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "4"),
            (2, "5"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
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
            (17, "5"),
            (20, "0"),
            (20, "5"),
            (21, "0"),
            (21, "5"),
            (22, "0"),
            (22, "5"),
            (23, "0"),
            (23, "2"),
            (23, "5"),
            (24, "5"),
            (25, "0"),
            (25, "2"),
            (25, "5"),
            (26, "0"),
            (26, "5"),
            (27, "0"),
            (27, "5"),
            (28, "0"),
            (28, "2"),
            (28, "5"),
            (29, "0"),
            (29, "5"),
        }:
            return 10
        elif key in {
            (2, "3"),
            (20, "2"),
            (21, "2"),
            (22, "1"),
            (22, "2"),
            (24, "0"),
            (24, "2"),
            (25, "1"),
            (26, "1"),
            (26, "2"),
            (27, "1"),
            (27, "2"),
            (28, "1"),
            (29, "1"),
            (29, "2"),
        }:
            return 6
        elif key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "<s>"),
            (3, "3"),
            (20, "3"),
            (21, "3"),
            (22, "3"),
            (23, "3"),
            (24, "3"),
            (25, "3"),
            (26, "3"),
            (27, "3"),
            (28, "3"),
            (29, "3"),
        }:
            return 13
        elif key in {(0, "0"), (0, "2"), (0, "3"), (0, "4"), (0, "5"), (0, "<s>")}:
            return 20
        return 22

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_7_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_2_output, attn_0_6_output):
        key = (attn_0_2_output, attn_0_6_output)
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
        return 10

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_6_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_0_output, attn_0_3_output):
        key = (attn_0_0_output, attn_0_3_output)
        return 3

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_3_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_7_output, num_attn_0_0_output):
        key = (num_attn_0_7_output, num_attn_0_0_output)
        return 1

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_1_output):
        key = (num_attn_0_3_output, num_attn_0_1_output)
        return 1

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_7_output, num_attn_0_6_output):
        key = (num_attn_0_7_output, num_attn_0_6_output)
        return 1

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_1_output, num_attn_0_4_output):
        key = (num_attn_0_1_output, num_attn_0_4_output)
        return 6

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_token, k_token):
        if q_token in {"4", "2", "1", "<s>", "0", "5"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "2"

    attn_1_0_pattern = select_closest(tokens, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1", "5", "4", "3"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == "1"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"0", "5", "2", "4"}:
            return k_token == ""
        elif q_token in {"1"}:
            return k_token == "<pad>"
        elif q_token in {"3"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_7_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_token, k_token):
        if q_token in {"1", "0", "5", "2"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "<s>"
        elif q_token in {"4"}:
            return k_token == "<pad>"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_1_3_pattern = select_closest(tokens, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0, 22}:
            return k_position == 15
        elif q_position in {1, 4, 5, 6, 23}:
            return k_position == 7
        elif q_position in {2}:
            return k_position == 13
        elif q_position in {3}:
            return k_position == 26
        elif q_position in {15, 7}:
            return k_position == 5
        elif q_position in {8, 11, 12, 13}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {16, 17, 19}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {21}:
            return k_position == 9
        elif q_position in {24, 25, 28}:
            return k_position == 1
        elif q_position in {26, 29}:
            return k_position == 22
        elif q_position in {27}:
            return k_position == 11

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_2_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1", "5", "4"}:
            return k_token == "0"
        elif q_token in {"2", "3"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == "2"

    attn_1_5_pattern = select_closest(tokens, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_5_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(token, position):
        if token in {"0"}:
            return position == 3
        elif token in {"1", "2", "3"}:
            return position == 11
        elif token in {"5", "<s>", "4"}:
            return position == 1

    attn_1_6_pattern = select_closest(positions, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_1_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, mlp_0_1_output):
        if position in {0}:
            return mlp_0_1_output == 13
        elif position in {1, 2, 3, 4, 7, 8, 9, 10, 14, 15, 16, 26}:
            return mlp_0_1_output == 22
        elif position in {5}:
            return mlp_0_1_output == 5
        elif position in {6}:
            return mlp_0_1_output == 1
        elif position in {11, 13, 18, 19, 28}:
            return mlp_0_1_output == 10
        elif position in {12, 20, 22}:
            return mlp_0_1_output == 20
        elif position in {17, 21, 23, 24, 25, 29}:
            return mlp_0_1_output == 12
        elif position in {27}:
            return mlp_0_1_output == 19

    attn_1_7_pattern = select_closest(mlp_0_1_outputs, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_6_output):
        if position in {0, 1, 20}:
            return attn_0_6_output == "2"
        elif position in {
            2,
            3,
            4,
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
            17,
            18,
            19,
            21,
            25,
        }:
            return attn_0_6_output == ""
        elif position in {8, 29}:
            return attn_0_6_output == "<s>"
        elif position in {22, 23, 24, 26, 28}:
            return attn_0_6_output == "5"
        elif position in {27}:
            return attn_0_6_output == "4"

    num_attn_1_0_pattern = select(attn_0_6_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(token, attn_0_0_output):
        if token in {"4", "2", "1", "0", "3"}:
            return attn_0_0_output == ""
        elif token in {"5"}:
            return attn_0_0_output == "5"
        elif token in {"<s>"}:
            return attn_0_0_output == "<pad>"

    num_attn_1_1_pattern = select(attn_0_0_outputs, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {"4", "1", "<s>", "0", "5", "3"}:
            return attn_0_1_output == ""
        elif attn_0_3_output in {"2"}:
            return attn_0_1_output == "2"

    num_attn_1_2_pattern = select(attn_0_1_outputs, attn_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_4_output):
        if position in {0, 27}:
            return attn_0_4_output == "2"
        elif position in {1, 20}:
            return attn_0_4_output == "3"
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
        }:
            return attn_0_4_output == ""
        elif position in {21, 22, 24, 25, 26, 28, 29}:
            return attn_0_4_output == "1"
        elif position in {23}:
            return attn_0_4_output == "4"

    num_attn_1_3_pattern = select(attn_0_4_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_4_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, token):
        if position in {0, 2, 3, 4, 5, 6, 21, 23, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {1, 20}:
            return token == "5"
        elif position in {7, 9, 10, 12, 14, 22}:
            return token == "4"
        elif position in {8, 19}:
            return token == "0"
        elif position in {16, 17, 11}:
            return token == "2"
        elif position in {13}:
            return token == "1"
        elif position in {18, 15}:
            return token == "3"
        elif position in {24}:
            return token == "<pad>"

    num_attn_1_4_pattern = select(tokens, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, ones)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, token):
        if position in {0, 16}:
            return token == "2"
        elif position in {1, 7, 10, 11, 12, 14}:
            return token == "4"
        elif position in {2, 3, 4, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {5, 6}:
            return token == "5"
        elif position in {8, 9, 13, 15}:
            return token == "0"
        elif position in {20}:
            return token == "3"

    num_attn_1_5_pattern = select(tokens, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, ones)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, attn_0_0_output):
        if position in {0, 24}:
            return attn_0_0_output == "2"
        elif position in {1, 2, 20}:
            return attn_0_0_output == "5"
        elif position in {
            3,
            4,
            7,
            9,
            10,
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
            26,
            29,
        }:
            return attn_0_0_output == ""
        elif position in {5, 6}:
            return attn_0_0_output == "3"
        elif position in {8, 11}:
            return attn_0_0_output == "<pad>"
        elif position in {25, 28, 23}:
            return attn_0_0_output == "1"
        elif position in {27}:
            return attn_0_0_output == "4"

    num_attn_1_6_pattern = select(attn_0_0_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_0_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            2,
            3,
            4,
            5,
            6,
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
        }:
            return token == ""
        elif mlp_0_0_output in {1, 29}:
            return token == "<s>"
        elif mlp_0_0_output in {7, 8, 9, 10, 11, 12, 14, 16}:
            return token == "1"

    num_attn_1_7_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, ones)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output, attn_1_7_output):
        key = (attn_1_0_output, attn_1_7_output)
        return 22

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_7_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_7_output, num_mlp_0_3_output):
        key = (attn_1_7_output, num_mlp_0_3_output)
        return 23

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_7_outputs, num_mlp_0_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(num_mlp_0_2_output, num_mlp_0_0_output):
        key = (num_mlp_0_2_output, num_mlp_0_0_output)
        return 26

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(num_mlp_0_2_outputs, num_mlp_0_0_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_6_output, attn_0_4_output):
        key = (attn_1_6_output, attn_0_4_output)
        return 8

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_0_4_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_7_output, num_attn_1_4_output):
        key = (num_attn_1_7_output, num_attn_1_4_output)
        return 8

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_7_output, num_attn_1_4_output):
        key = (num_attn_1_7_output, num_attn_1_4_output)
        return 4

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_4_output, num_attn_1_0_output):
        key = (num_attn_1_4_output, num_attn_1_0_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
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
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
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
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
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
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
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
            (4, 7),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
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
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (5, 20),
            (5, 21),
            (5, 22),
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
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
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
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (7, 19),
            (7, 20),
            (7, 21),
            (7, 22),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 28),
            (7, 29),
            (7, 30),
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
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (8, 21),
            (8, 22),
            (8, 23),
            (8, 24),
            (8, 25),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (8, 30),
            (8, 31),
            (8, 32),
            (8, 33),
            (8, 34),
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
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 23),
            (9, 24),
            (9, 25),
            (9, 26),
            (9, 27),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 34),
            (9, 35),
            (9, 36),
            (9, 37),
            (9, 38),
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
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (10, 23),
            (10, 24),
            (10, 25),
            (10, 26),
            (10, 27),
            (10, 28),
            (10, 29),
            (10, 30),
            (10, 31),
            (10, 32),
            (10, 33),
            (10, 34),
            (10, 35),
            (10, 36),
            (10, 37),
            (10, 38),
            (10, 39),
            (10, 40),
            (10, 41),
            (10, 42),
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
            (11, 21),
            (11, 22),
            (11, 23),
            (11, 24),
            (11, 25),
            (11, 26),
            (11, 27),
            (11, 28),
            (11, 29),
            (11, 30),
            (11, 31),
            (11, 32),
            (11, 33),
            (11, 34),
            (11, 35),
            (11, 36),
            (11, 37),
            (11, 38),
            (11, 39),
            (11, 40),
            (11, 41),
            (11, 42),
            (11, 43),
            (11, 44),
            (11, 45),
            (11, 46),
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
            (12, 23),
            (12, 24),
            (12, 25),
            (12, 26),
            (12, 27),
            (12, 28),
            (12, 29),
            (12, 30),
            (12, 31),
            (12, 32),
            (12, 33),
            (12, 34),
            (12, 35),
            (12, 36),
            (12, 37),
            (12, 38),
            (12, 39),
            (12, 40),
            (12, 41),
            (12, 42),
            (12, 43),
            (12, 44),
            (12, 45),
            (12, 46),
            (12, 47),
            (12, 48),
            (12, 49),
            (12, 50),
            (12, 51),
            (12, 52),
            (12, 53),
            (12, 54),
            (12, 55),
            (12, 56),
            (12, 57),
            (12, 58),
            (12, 59),
            (13, 25),
            (13, 26),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 30),
            (13, 31),
            (13, 32),
            (13, 33),
            (13, 34),
            (13, 35),
            (13, 36),
            (13, 37),
            (13, 38),
            (13, 39),
            (13, 40),
            (13, 41),
            (13, 42),
            (13, 43),
            (13, 44),
            (13, 45),
            (13, 46),
            (13, 47),
            (13, 48),
            (13, 49),
            (13, 50),
            (13, 51),
            (13, 52),
            (13, 53),
            (13, 54),
            (13, 55),
            (13, 56),
            (13, 57),
            (13, 58),
            (13, 59),
            (14, 27),
            (14, 28),
            (14, 29),
            (14, 30),
            (14, 31),
            (14, 32),
            (14, 33),
            (14, 34),
            (14, 35),
            (14, 36),
            (14, 37),
            (14, 38),
            (14, 39),
            (14, 40),
            (14, 41),
            (14, 42),
            (14, 43),
            (14, 44),
            (14, 45),
            (14, 46),
            (14, 47),
            (14, 48),
            (14, 49),
            (14, 50),
            (14, 51),
            (14, 52),
            (14, 53),
            (14, 54),
            (14, 55),
            (14, 56),
            (14, 57),
            (14, 58),
            (14, 59),
            (15, 29),
            (15, 30),
            (15, 31),
            (15, 32),
            (15, 33),
            (15, 34),
            (15, 35),
            (15, 36),
            (15, 37),
            (15, 38),
            (15, 39),
            (15, 40),
            (15, 41),
            (15, 42),
            (15, 43),
            (15, 44),
            (15, 45),
            (15, 46),
            (15, 47),
            (15, 48),
            (15, 49),
            (15, 50),
            (15, 51),
            (15, 52),
            (15, 53),
            (15, 54),
            (15, 55),
            (15, 56),
            (15, 57),
            (15, 58),
            (15, 59),
            (16, 31),
            (16, 32),
            (16, 33),
            (16, 34),
            (16, 35),
            (16, 36),
            (16, 37),
            (16, 38),
            (16, 39),
            (16, 40),
            (16, 41),
            (16, 42),
            (16, 43),
            (16, 44),
            (16, 45),
            (16, 46),
            (16, 47),
            (16, 48),
            (16, 49),
            (16, 50),
            (16, 51),
            (16, 52),
            (16, 53),
            (16, 54),
            (16, 55),
            (16, 56),
            (16, 57),
            (16, 58),
            (16, 59),
            (17, 33),
            (17, 34),
            (17, 35),
            (17, 36),
            (17, 37),
            (17, 38),
            (17, 39),
            (17, 40),
            (17, 41),
            (17, 42),
            (17, 43),
            (17, 44),
            (17, 45),
            (17, 46),
            (17, 47),
            (17, 48),
            (17, 49),
            (17, 50),
            (17, 51),
            (17, 52),
            (17, 53),
            (17, 54),
            (17, 55),
            (17, 56),
            (17, 57),
            (17, 58),
            (17, 59),
            (18, 35),
            (18, 36),
            (18, 37),
            (18, 38),
            (18, 39),
            (18, 40),
            (18, 41),
            (18, 42),
            (18, 43),
            (18, 44),
            (18, 45),
            (18, 46),
            (18, 47),
            (18, 48),
            (18, 49),
            (18, 50),
            (18, 51),
            (18, 52),
            (18, 53),
            (18, 54),
            (18, 55),
            (18, 56),
            (18, 57),
            (18, 58),
            (18, 59),
            (19, 37),
            (19, 38),
            (19, 39),
            (19, 40),
            (19, 41),
            (19, 42),
            (19, 43),
            (19, 44),
            (19, 45),
            (19, 46),
            (19, 47),
            (19, 48),
            (19, 49),
            (19, 50),
            (19, 51),
            (19, 52),
            (19, 53),
            (19, 54),
            (19, 55),
            (19, 56),
            (19, 57),
            (19, 58),
            (19, 59),
            (20, 39),
            (20, 40),
            (20, 41),
            (20, 42),
            (20, 43),
            (20, 44),
            (20, 45),
            (20, 46),
            (20, 47),
            (20, 48),
            (20, 49),
            (20, 50),
            (20, 51),
            (20, 52),
            (20, 53),
            (20, 54),
            (20, 55),
            (20, 56),
            (20, 57),
            (20, 58),
            (20, 59),
            (21, 41),
            (21, 42),
            (21, 43),
            (21, 44),
            (21, 45),
            (21, 46),
            (21, 47),
            (21, 48),
            (21, 49),
            (21, 50),
            (21, 51),
            (21, 52),
            (21, 53),
            (21, 54),
            (21, 55),
            (21, 56),
            (21, 57),
            (21, 58),
            (21, 59),
            (22, 43),
            (22, 44),
            (22, 45),
            (22, 46),
            (22, 47),
            (22, 48),
            (22, 49),
            (22, 50),
            (22, 51),
            (22, 52),
            (22, 53),
            (22, 54),
            (22, 55),
            (22, 56),
            (22, 57),
            (22, 58),
            (22, 59),
            (23, 45),
            (23, 46),
            (23, 47),
            (23, 48),
            (23, 49),
            (23, 50),
            (23, 51),
            (23, 52),
            (23, 53),
            (23, 54),
            (23, 55),
            (23, 56),
            (23, 57),
            (23, 58),
            (23, 59),
            (24, 47),
            (24, 48),
            (24, 49),
            (24, 50),
            (24, 51),
            (24, 52),
            (24, 53),
            (24, 54),
            (24, 55),
            (24, 56),
            (24, 57),
            (24, 58),
            (24, 59),
            (25, 49),
            (25, 50),
            (25, 51),
            (25, 52),
            (25, 53),
            (25, 54),
            (25, 55),
            (25, 56),
            (25, 57),
            (25, 58),
            (25, 59),
            (26, 51),
            (26, 52),
            (26, 53),
            (26, 54),
            (26, 55),
            (26, 56),
            (26, 57),
            (26, 58),
            (26, 59),
            (27, 53),
            (27, 54),
            (27, 55),
            (27, 56),
            (27, 57),
            (27, 58),
            (27, 59),
            (28, 55),
            (28, 56),
            (28, 57),
            (28, 58),
            (28, 59),
            (29, 57),
            (29, 58),
            (29, 59),
            (30, 59),
        }:
            return 8
        return 17

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_3_output, num_attn_1_6_output):
        key = (num_attn_1_3_output, num_attn_1_6_output)
        return 27

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"1", "0"}:
            return position == 9
        elif token in {"2", "3"}:
            return position == 10
        elif token in {"<s>", "4"}:
            return position == 7
        elif token in {"5"}:
            return position == 12

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_7_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_2_output, attn_0_0_output):
        if attn_1_2_output in {"0", "3"}:
            return attn_0_0_output == "4"
        elif attn_1_2_output in {"1"}:
            return attn_0_0_output == "<s>"
        elif attn_1_2_output in {"<s>", "2", "4"}:
            return attn_0_0_output == ""
        elif attn_1_2_output in {"5"}:
            return attn_0_0_output == "3"

    attn_2_1_pattern = select_closest(attn_0_0_outputs, attn_1_2_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, mlp_0_0_output):
        if token in {"0"}:
            return mlp_0_0_output == 0
        elif token in {"1"}:
            return mlp_0_0_output == 1
        elif token in {"2"}:
            return mlp_0_0_output == 11
        elif token in {"3"}:
            return mlp_0_0_output == 12
        elif token in {"4"}:
            return mlp_0_0_output == 13
        elif token in {"5"}:
            return mlp_0_0_output == 7
        elif token in {"<s>"}:
            return mlp_0_0_output == 18

    attn_2_2_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_6_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_3_output, token):
        if attn_0_3_output in {"1", "0", "5", "<s>"}:
            return token == "3"
        elif attn_0_3_output in {"2"}:
            return token == "1"
        elif attn_0_3_output in {"3"}:
            return token == ""
        elif attn_0_3_output in {"4"}:
            return token == "2"

    attn_2_3_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "0"
        elif attn_0_4_output in {"1"}:
            return token == ""
        elif attn_0_4_output in {"2", "3"}:
            return token == "2"
        elif attn_0_4_output in {"4"}:
            return token == "1"
        elif attn_0_4_output in {"5"}:
            return token == "3"
        elif attn_0_4_output in {"<s>"}:
            return token == "<s>"

    attn_2_4_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_0_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(token, position):
        if token in {"0"}:
            return position == 5
        elif token in {"1"}:
            return position == 1
        elif token in {"2"}:
            return position == 6
        elif token in {"3"}:
            return position == 22
        elif token in {"4"}:
            return position == 7
        elif token in {"5"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 13

    attn_2_5_pattern = select_closest(positions, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, mlp_0_0_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
        if q_token in {"0", "5"}:
            return k_token == "2"
        elif q_token in {"1", "4"}:
            return k_token == "<s>"
        elif q_token in {"2"}:
            return k_token == "5"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_5_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 22}:
            return token == "0"
        elif mlp_0_0_output in {1, 2, 4, 13, 18, 21, 25}:
            return token == "<s>"
        elif mlp_0_0_output in {3}:
            return token == "3"
        elif mlp_0_0_output in {5, 6, 7, 12, 15, 16, 17, 19, 23, 24, 26, 27, 28, 29}:
            return token == ""
        elif mlp_0_0_output in {8, 11, 14}:
            return token == "4"
        elif mlp_0_0_output in {9}:
            return token == "5"
        elif mlp_0_0_output in {10}:
            return token == "<pad>"
        elif mlp_0_0_output in {20}:
            return token == "2"

    attn_2_7_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_6_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, attn_0_3_output):
        if attn_1_2_output in {"4", "2", "1", "<s>", "0", "5"}:
            return attn_0_3_output == ""
        elif attn_1_2_output in {"3"}:
            return attn_0_3_output == "3"

    num_attn_2_0_pattern = select(attn_0_3_outputs, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_2_output, attn_0_1_output):
        if attn_0_2_output in {"0"}:
            return attn_0_1_output == "0"
        elif attn_0_2_output in {"4", "1", "<s>", "5", "3"}:
            return attn_0_1_output == ""
        elif attn_0_2_output in {"2"}:
            return attn_0_1_output == "<s>"

    num_attn_2_1_pattern = select(attn_0_1_outputs, attn_0_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(token, attn_1_1_output):
        if token in {"4", "2", "<s>", "0", "5", "3"}:
            return attn_1_1_output == ""
        elif token in {"1"}:
            return attn_1_1_output == "1"

    num_attn_2_2_pattern = select(attn_1_1_outputs, tokens, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_6_output, attn_1_5_output):
        if attn_0_6_output in {"0"}:
            return attn_1_5_output == 26
        elif attn_0_6_output in {"1", "3"}:
            return attn_1_5_output == 23
        elif attn_0_6_output in {"<s>", "2"}:
            return attn_1_5_output == 4
        elif attn_0_6_output in {"4"}:
            return attn_1_5_output == 22
        elif attn_0_6_output in {"5"}:
            return attn_1_5_output == 28

    num_attn_2_3_pattern = select(attn_1_5_outputs, attn_0_6_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(token, attn_1_7_output):
        if token in {"4", "2", "1", "<s>", "0", "5"}:
            return attn_1_7_output == ""
        elif token in {"3"}:
            return attn_1_7_output == "3"

    num_attn_2_4_pattern = select(attn_1_7_outputs, tokens, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_4_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(position, token):
        if position in {0, 21}:
            return token == "3"
        elif position in {1, 20}:
            return token == "4"
        elif position in {
            2,
            3,
            4,
            5,
            6,
            9,
            11,
            13,
            14,
            17,
            18,
            19,
            22,
            23,
            24,
            25,
            28,
            29,
        }:
            return token == ""
        elif position in {7, 8, 10, 12, 15}:
            return token == "<s>"
        elif position in {16}:
            return token == "<pad>"
        elif position in {26}:
            return token == "0"
        elif position in {27}:
            return token == "1"

    num_attn_2_5_pattern = select(tokens, positions, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, ones)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_4_output, attn_0_1_output):
        if attn_0_4_output in {"4", "2", "1", "0", "5"}:
            return attn_0_1_output == ""
        elif attn_0_4_output in {"3"}:
            return attn_0_1_output == "3"
        elif attn_0_4_output in {"<s>"}:
            return attn_0_1_output == "<pad>"

    num_attn_2_6_pattern = select(attn_0_1_outputs, attn_0_4_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_4_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_4_output, attn_1_3_output):
        if attn_0_4_output in {"1", "0", "5", "2"}:
            return attn_1_3_output == "4"
        elif attn_0_4_output in {"3"}:
            return attn_1_3_output == ""
        elif attn_0_4_output in {"4"}:
            return attn_1_3_output == "5"
        elif attn_0_4_output in {"<s>"}:
            return attn_1_3_output == "3"

    num_attn_2_7_pattern = select(attn_1_3_outputs, attn_0_4_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_4_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, attn_1_4_output):
        key = (attn_2_1_output, attn_1_4_output)
        return 9

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_1_4_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(mlp_1_3_output, attn_2_7_output):
        key = (mlp_1_3_output, attn_2_7_output)
        return 25

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(mlp_1_3_outputs, attn_2_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_1_5_output, mlp_0_1_output):
        key = (attn_1_5_output, mlp_0_1_output)
        return 16

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_1_5_outputs, mlp_0_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_1_4_output, attn_1_3_output):
        key = (attn_1_4_output, attn_1_3_output)
        if key in {
            ("0", "4"),
            ("1", "4"),
            ("2", "4"),
            ("3", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "<s>"),
            ("5", "4"),
            ("<s>", "4"),
        }:
            return 19
        return 3

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_1_4_outputs, attn_1_3_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_0_output, num_attn_2_2_output):
        key = (num_attn_2_0_output, num_attn_2_2_output)
        return 22

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_5_output, num_attn_2_6_output):
        key = (num_attn_2_5_output, num_attn_2_6_output)
        return 28

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_5_outputs, num_attn_2_6_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_1_output, num_attn_1_7_output):
        key = (num_attn_1_1_output, num_attn_1_7_output)
        return 8

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_7_output, num_attn_1_5_output):
        key = (num_attn_2_7_output, num_attn_1_5_output)
        return 20

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_1_5_outputs)
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


print(run(["<s>", "0", "1", "3", "0", "0", "0", "5", "5", "3", "2", "3"]))
