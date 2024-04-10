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
        "output/length/rasp/mostfreq/trainlength10/s2/most_freq_weights.csv",
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
        if q_position in {0, 1, 7}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {9, 4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {10}:
            return k_position == 17
        elif q_position in {17, 11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {16, 14}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 11
        elif q_position in {19}:
            return k_position == 13

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {8, 9, 4}:
            return k_position == 0
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {10, 7}:
            return k_position == 3
        elif q_position in {16, 11, 15}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {13, 14}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 16

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"4", "0", "3", "1", "5", "2"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "4"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0}:
            return token == "4"
        elif position in {1}:
            return token == "2"
        elif position in {2, 3, 4, 5}:
            return token == "1"
        elif position in {6}:
            return token == "0"
        elif position in {7}:
            return token == "5"
        elif position in {8}:
            return token == "3"
        elif position in {9}:
            return token == "<s>"
        elif position in {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1, 2, 5}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {8}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 3
        elif q_position in {16, 10}:
            return k_position == 11
        elif q_position in {19, 18, 11}:
            return k_position == 18
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {17, 13}:
            return k_position == 17
        elif q_position in {14}:
            return k_position == 14
        elif q_position in {15}:
            return k_position == 15

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3, 7}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 0
        elif q_position in {9}:
            return k_position == 6
        elif q_position in {17, 10, 18}:
            return k_position == 15
        elif q_position in {11, 13}:
            return k_position == 17
        elif q_position in {12, 15}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 11
        elif q_position in {16, 19}:
            return k_position == 16

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 7}:
            return k_position == 2
        elif q_position in {3, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {8, 6}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 3
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {16, 11}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 16
        elif q_position in {17, 13, 15}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {18}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 18

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0}:
            return token == "3"
        elif position in {8, 1}:
            return token == "0"
        elif position in {2, 3, 4, 6}:
            return token == "2"
        elif position in {9, 5}:
            return token == "1"
        elif position in {7}:
            return token == "4"
        elif position in {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1}:
            return token == "4"
        elif position in {2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""
        elif position in {9, 6}:
            return token == "<s>"
        elif position in {7}:
            return token == "2"
        elif position in {8}:
            return token == "0"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 1, 7}:
            return token == "1"
        elif position in {2}:
            return token == "<pad>"
        elif position in {3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""
        elif position in {5, 6}:
            return token == "<s>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""
        elif position in {1}:
            return token == "0"
        elif position in {4, 5, 6}:
            return token == "5"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""
        elif position in {1}:
            return token == "5"
        elif position in {8, 9, 5, 6}:
            return token == "<s>"
        elif position in {7}:
            return token == "4"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0, 6, 7}:
            return k_position == 4
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {8, 3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {18, 12, 13}:
            return k_position == 10
        elif q_position in {19, 14}:
            return k_position == 3
        elif q_position in {16, 15}:
            return k_position == 18
        elif q_position in {17}:
            return k_position == 8

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""
        elif position in {1}:
            return token == "0"
        elif position in {5, 6, 7}:
            return token == "<s>"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 1, 2}:
            return token == "1"
        elif position in {3, 4}:
            return token == "<s>"
        elif position in {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19}:
            return token == ""
        elif position in {16}:
            return token == "0"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 1, 10, 13, 16}:
            return token == "2"
        elif position in {2, 3, 4, 5, 9, 11, 12, 14, 15, 18, 19}:
            return token == ""
        elif position in {8, 17, 6, 7}:
            return token == "<s>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position, attn_0_7_output):
        key = (position, attn_0_7_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "<s>"),
            (3, "0"),
            (3, "3"),
            (11, "0"),
            (14, "0"),
            (15, "0"),
        }:
            return 6
        elif key in {
            (0, "2"),
            (2, "2"),
            (3, "2"),
            (4, "2"),
            (5, "2"),
            (6, "2"),
            (7, "2"),
            (8, "0"),
            (8, "2"),
            (8, "3"),
            (8, "5"),
            (8, "<s>"),
            (9, "0"),
            (9, "2"),
            (10, "2"),
            (11, "2"),
            (12, "2"),
            (13, "2"),
            (14, "2"),
            (15, "2"),
            (16, "2"),
            (17, "2"),
            (18, "2"),
            (19, "2"),
        }:
            return 2
        elif key in {
            (0, "0"),
            (0, "3"),
            (5, "0"),
            (5, "3"),
            (6, "0"),
            (6, "3"),
            (10, "3"),
            (11, "3"),
            (12, "3"),
            (14, "3"),
            (15, "3"),
            (16, "3"),
            (17, "3"),
            (18, "3"),
            (19, "3"),
        }:
            return 19
        return 0

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(positions, attn_0_7_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_4_output, position):
        key = (attn_0_4_output, position)
        if key in {
            ("1", 0),
            ("1", 7),
            ("1", 8),
            ("1", 9),
            ("2", 0),
            ("2", 7),
            ("2", 8),
            ("2", 9),
            ("3", 0),
            ("3", 7),
            ("3", 8),
            ("3", 9),
            ("4", 0),
            ("4", 7),
            ("4", 8),
            ("4", 9),
            ("5", 0),
            ("5", 7),
            ("5", 8),
            ("<s>", 0),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 13),
            ("<s>", 15),
            ("<s>", 16),
        }:
            return 4
        elif key in {
            ("2", 2),
            ("3", 2),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 13),
            ("3", 14),
            ("3", 15),
            ("3", 16),
            ("3", 17),
            ("3", 18),
            ("3", 19),
            ("4", 2),
            ("4", 6),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 14),
            ("4", 15),
            ("4", 16),
            ("4", 17),
            ("4", 18),
            ("4", 19),
            ("5", 2),
            ("5", 6),
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
            ("<s>", 2),
        }:
            return 0
        elif key in {
            ("0", 1),
            ("0", 7),
            ("1", 1),
            ("1", 2),
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
            ("3", 1),
            ("5", 1),
            ("<s>", 1),
        }:
            return 6
        elif key in {
            ("2", 3),
            ("2", 4),
            ("2", 5),
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
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 14),
            ("<s>", 17),
            ("<s>", 19),
        }:
            return 13
        elif key in {("0", 8), ("5", 3), ("5", 4), ("5", 5), ("5", 9)}:
            return 10
        elif key in {("0", 6), ("1", 6), ("2", 6), ("3", 6), ("<s>", 6)}:
            return 15
        elif key in {("0", 0), ("2", 1), ("3", 3), ("3", 4), ("3", 5)}:
            return 14
        elif key in {
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("0", 13),
            ("0", 15),
            ("0", 18),
            ("0", 19),
        }:
            return 16
        elif key in {("<s>", 11), ("<s>", 12), ("<s>", 18)}:
            return 19
        elif key in {("4", 1)}:
            return 12
        elif key in {("0", 9)}:
            return 18
        return 9

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_4_outputs, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(position, attn_0_2_output):
        key = (position, attn_0_2_output)
        if key in {
            (1, "5"),
            (10, "5"),
            (11, "5"),
            (12, "5"),
            (14, "5"),
            (15, "5"),
            (16, "5"),
            (17, "5"),
            (18, "5"),
            (19, "5"),
        }:
            return 7
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "<s>"),
            (10, "3"),
            (15, "3"),
            (17, "3"),
            (19, "3"),
        }:
            return 19
        elif key in {
            (0, "<s>"),
            (4, "<s>"),
            (10, "0"),
            (10, "1"),
            (10, "2"),
            (11, "0"),
            (11, "1"),
            (11, "2"),
            (11, "3"),
            (12, "0"),
            (12, "1"),
            (12, "2"),
            (12, "3"),
            (13, "0"),
            (13, "1"),
            (13, "2"),
            (13, "3"),
            (14, "0"),
            (14, "1"),
            (14, "2"),
            (14, "3"),
            (15, "0"),
            (15, "1"),
            (15, "2"),
            (16, "0"),
            (16, "1"),
            (16, "2"),
            (16, "3"),
            (17, "0"),
            (17, "1"),
            (17, "2"),
            (18, "0"),
            (18, "1"),
            (18, "2"),
            (18, "3"),
            (19, "0"),
            (19, "1"),
            (19, "2"),
            (19, "<s>"),
        }:
            return 12
        elif key in {
            (1, "4"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "<s>"),
        }:
            return 10
        elif key in {
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "<s>"),
        }:
            return 1
        elif key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "<s>"),
            (10, "<s>"),
            (11, "<s>"),
            (12, "<s>"),
            (13, "<s>"),
            (14, "<s>"),
            (15, "<s>"),
            (16, "<s>"),
            (17, "<s>"),
            (18, "<s>"),
        }:
            return 4
        elif key in {(0, "5"), (2, "5"), (3, "5"), (4, "5"), (7, "5")}:
            return 17
        elif key in {(13, "5")}:
            return 14
        return 8

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(positions, attn_0_2_outputs)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_7_output, position):
        key = (attn_0_7_output, position)
        if key in {
            ("0", 6),
            ("0", 7),
            ("0", 8),
            ("0", 9),
            ("1", 6),
            ("1", 7),
            ("1", 8),
            ("1", 9),
            ("2", 6),
            ("2", 7),
            ("2", 8),
            ("2", 9),
            ("3", 0),
            ("3", 6),
            ("3", 7),
            ("3", 8),
            ("3", 9),
            ("4", 6),
            ("4", 7),
            ("4", 8),
            ("4", 9),
            ("5", 6),
            ("5", 7),
            ("5", 8),
            ("5", 9),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 9),
        }:
            return 16
        elif key in {
            ("0", 2),
            ("0", 3),
            ("2", 1),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 5),
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
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 13),
            ("3", 14),
            ("3", 15),
            ("3", 16),
            ("3", 17),
            ("3", 18),
            ("3", 19),
            ("4", 2),
            ("5", 2),
            ("5", 3),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
        }:
            return 10
        elif key in {
            ("0", 0),
            ("0", 12),
            ("0", 13),
            ("0", 15),
            ("0", 16),
            ("0", 18),
            ("0", 19),
            ("1", 0),
            ("1", 15),
            ("2", 0),
            ("4", 0),
            ("5", 0),
            ("5", 12),
            ("5", 13),
            ("5", 15),
            ("5", 19),
            ("<s>", 0),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 15),
            ("<s>", 16),
            ("<s>", 17),
            ("<s>", 18),
            ("<s>", 19),
        }:
            return 18
        elif key in {("0", 1), ("4", 1), ("5", 1), ("<s>", 1)}:
            return 14
        elif key in {("3", 4), ("3", 5)}:
            return 17
        return 4

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_7_outputs, positions)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output, num_attn_0_3_output):
        key = (num_attn_0_5_output, num_attn_0_3_output)
        return 3

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_4_output):
        key = (num_attn_0_3_output, num_attn_0_4_output)
        return 19

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_5_output):
        key = num_attn_0_5_output
        if key in {0, 1}:
            return 3
        return 1

    num_mlp_0_2_outputs = [num_mlp_0_2(k0) for k0 in num_attn_0_5_outputs]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_0_output, num_attn_0_7_output):
        key = (num_attn_0_0_output, num_attn_0_7_output)
        return 5

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 3, 5, 6}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {4, 7}:
            return k_position == 6
        elif q_position in {8, 9}:
            return k_position == 7
        elif q_position in {10, 12, 14}:
            return k_position == 17
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {16, 17, 13}:
            return k_position == 0
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {18}:
            return k_position == 11
        elif q_position in {19}:
            return k_position == 16

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_0_output, position):
        if attn_0_0_output in {"<s>", "0", "3"}:
            return position == 0
        elif attn_0_0_output in {"4", "5", "1"}:
            return position == 4
        elif attn_0_0_output in {"2"}:
            return position == 2

    attn_1_1_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_3_output, mlp_0_2_output):
        if attn_0_3_output in {"0"}:
            return mlp_0_2_output == 0
        elif attn_0_3_output in {"1"}:
            return mlp_0_2_output == 4
        elif attn_0_3_output in {"2"}:
            return mlp_0_2_output == 15
        elif attn_0_3_output in {"3"}:
            return mlp_0_2_output == 8
        elif attn_0_3_output in {"4"}:
            return mlp_0_2_output == 9
        elif attn_0_3_output in {"<s>", "5"}:
            return mlp_0_2_output == 1

    attn_1_2_pattern = select_closest(mlp_0_2_outputs, attn_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_6_output, token):
        if attn_0_6_output in {"<s>", "0", "1"}:
            return token == "<s>"
        elif attn_0_6_output in {"4", "5", "2"}:
            return token == "3"
        elif attn_0_6_output in {"3"}:
            return token == "0"

    attn_1_3_pattern = select_closest(tokens, attn_0_6_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(token, mlp_0_1_output):
        if token in {"0", "3", "2"}:
            return mlp_0_1_output == 7
        elif token in {"1"}:
            return mlp_0_1_output == 8
        elif token in {"4"}:
            return mlp_0_1_output == 12
        elif token in {"5"}:
            return mlp_0_1_output == 15
        elif token in {"<s>"}:
            return mlp_0_1_output == 0

    attn_1_4_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, mlp_0_2_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_6_output, mlp_0_2_output):
        if attn_0_6_output in {"0"}:
            return mlp_0_2_output == 18
        elif attn_0_6_output in {"1"}:
            return mlp_0_2_output == 11
        elif attn_0_6_output in {"3", "2"}:
            return mlp_0_2_output == 17
        elif attn_0_6_output in {"4", "<s>"}:
            return mlp_0_2_output == 5
        elif attn_0_6_output in {"5"}:
            return mlp_0_2_output == 10

    attn_1_5_pattern = select_closest(mlp_0_2_outputs, attn_0_6_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, mlp_0_2_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 5, 6}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {8, 9, 4}:
            return k_position == 9
        elif q_position in {18, 7}:
            return k_position == 8
        elif q_position in {10, 11, 12, 13, 14, 15, 17}:
            return k_position == 12
        elif q_position in {16}:
            return k_position == 0
        elif q_position in {19}:
            return k_position == 10

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_1_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(token, mlp_0_1_output):
        if token in {"<s>", "0", "5", "2"}:
            return mlp_0_1_output == 7
        elif token in {"1"}:
            return mlp_0_1_output == 2
        elif token in {"3"}:
            return mlp_0_1_output == 1
        elif token in {"4"}:
            return mlp_0_1_output == 12

    attn_1_7_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_0_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_3_output, attn_0_7_output):
        if attn_0_3_output in {"4", "<s>", "0", "3", "5", "2"}:
            return attn_0_7_output == "1"
        elif attn_0_3_output in {"1"}:
            return attn_0_7_output == "3"

    num_attn_1_0_pattern = select(attn_0_7_outputs, attn_0_3_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_5_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"0", "3", "5", "1", "2"}:
            return k_attn_0_1_output == "4"
        elif q_attn_0_1_output in {"4"}:
            return k_attn_0_1_output == "0"
        elif q_attn_0_1_output in {"<s>"}:
            return k_attn_0_1_output == ""

    num_attn_1_1_pattern = select(attn_0_1_outputs, attn_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_6_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, attn_0_5_output):
        if attn_0_1_output in {"4", "0", "3", "5", "2"}:
            return attn_0_5_output == ""
        elif attn_0_1_output in {"<s>", "1"}:
            return attn_0_5_output == "1"

    num_attn_1_2_pattern = select(attn_0_5_outputs, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_4_output, attn_0_5_output):
        if attn_0_4_output in {"0", "2", "1"}:
            return attn_0_5_output == "3"
        elif attn_0_4_output in {"3"}:
            return attn_0_5_output == "2"
        elif attn_0_4_output in {"4"}:
            return attn_0_5_output == ""
        elif attn_0_4_output in {"5"}:
            return attn_0_5_output == "<s>"
        elif attn_0_4_output in {"<s>"}:
            return attn_0_5_output == "1"

    num_attn_1_3_pattern = select(attn_0_5_outputs, attn_0_4_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_1_output, num_mlp_0_2_output):
        if attn_0_1_output in {"<s>", "0"}:
            return num_mlp_0_2_output == 1
        elif attn_0_1_output in {"1"}:
            return num_mlp_0_2_output == 15
        elif attn_0_1_output in {"4", "3", "2"}:
            return num_mlp_0_2_output == 14
        elif attn_0_1_output in {"5"}:
            return num_mlp_0_2_output == 9

    num_attn_1_4_pattern = select(
        num_mlp_0_2_outputs, attn_0_1_outputs, num_predicate_1_4
    )
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_5_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_3_output, attn_0_0_output):
        if attn_0_3_output in {"0"}:
            return attn_0_0_output == "1"
        elif attn_0_3_output in {"3", "1"}:
            return attn_0_0_output == ""
        elif attn_0_3_output in {"4", "5", "2"}:
            return attn_0_0_output == "0"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_0_output == "<pad>"

    num_attn_1_5_pattern = select(attn_0_0_outputs, attn_0_3_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_3_output, attn_0_5_output):
        if attn_0_3_output in {"0", "1"}:
            return attn_0_5_output == ""
        elif attn_0_3_output in {"4", "5", "<s>", "2"}:
            return attn_0_5_output == "3"
        elif attn_0_3_output in {"3"}:
            return attn_0_5_output == "0"

    num_attn_1_6_pattern = select(attn_0_5_outputs, attn_0_3_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_6_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1, 2, 3, 10, 12, 13, 14, 16, 17, 18, 19}:
            return token == ""
        elif position in {4, 5, 6}:
            return token == "4"
        elif position in {7}:
            return token == "2"
        elif position in {8, 9}:
            return token == "1"
        elif position in {11, 15}:
            return token == "<pad>"

    num_attn_1_7_pattern = select(tokens, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "5"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("2", "0"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "5"),
            ("4", "0"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("5", "0"),
            ("5", "2"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "5"),
        }:
            return 15
        return 9

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_7_output, num_mlp_0_0_output):
        key = (attn_1_7_output, num_mlp_0_0_output)
        return 3

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_7_outputs, num_mlp_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_1_output):
        key = attn_1_1_output
        if key in {10, 12, 14, 19}:
            return 8
        return 5

    mlp_1_2_outputs = [mlp_1_2(k0) for k0 in attn_1_1_outputs]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_0_0_output, attn_1_3_output):
        key = (attn_0_0_output, attn_1_3_output)
        return 8

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_1_3_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_1_4_output):
        key = (num_attn_1_2_output, num_attn_1_4_output)
        return 12

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_7_output):
        key = num_attn_1_7_output
        return 11

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_1_7_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_7_output, num_attn_1_2_output):
        key = (num_attn_1_7_output, num_attn_1_2_output)
        return 17

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_7_output, num_attn_1_0_output):
        key = (num_attn_1_7_output, num_attn_1_0_output)
        return 10

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 16
        elif q_position in {1, 5, 6}:
            return k_position == 3
        elif q_position in {2, 3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {8, 9, 7}:
            return k_position == 1
        elif q_position in {16, 18, 11, 15}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 0
        elif q_position in {14}:
            return k_position == 11
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {19}:
            return k_position == 14

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 18, 19}:
            return token == "<s>"
        elif position in {1, 5, 6}:
            return token == "3"
        elif position in {8, 7}:
            return token == "1"
        elif position in {9}:
            return token == "4"
        elif position in {17}:
            return token == "<pad>"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_4_output, mlp_0_3_output):
        if attn_0_4_output in {"0", "3"}:
            return mlp_0_3_output == 10
        elif attn_0_4_output in {"1"}:
            return mlp_0_3_output == 4
        elif attn_0_4_output in {"2"}:
            return mlp_0_3_output == 7
        elif attn_0_4_output in {"4", "5"}:
            return mlp_0_3_output == 19
        elif attn_0_4_output in {"<s>"}:
            return mlp_0_3_output == 12

    attn_2_2_pattern = select_closest(mlp_0_3_outputs, attn_0_4_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_6_output, position):
        if attn_0_6_output in {"0"}:
            return position == 18
        elif attn_0_6_output in {"<s>", "3", "5", "1"}:
            return position == 0
        elif attn_0_6_output in {"2"}:
            return position == 13
        elif attn_0_6_output in {"4"}:
            return position == 7

    attn_2_3_pattern = select_closest(positions, attn_0_6_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_0_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_position, k_position):
        if q_position in {0, 10, 13, 18, 19}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3, 4, 6}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {8, 9, 7}:
            return k_position == 1
        elif q_position in {11}:
            return k_position == 11
        elif q_position in {16, 12, 15}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 0
        elif q_position in {17}:
            return k_position == 13

    attn_2_4_pattern = select_closest(positions, positions, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, mlp_0_3_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(position, mlp_0_3_output):
        if position in {0, 2, 10, 12, 13, 14, 15, 16, 18, 19}:
            return mlp_0_3_output == 17
        elif position in {1}:
            return mlp_0_3_output == 0
        elif position in {3, 4}:
            return mlp_0_3_output == 11
        elif position in {5, 6, 7}:
            return mlp_0_3_output == 14
        elif position in {8, 9}:
            return mlp_0_3_output == 1
        elif position in {11}:
            return mlp_0_3_output == 4
        elif position in {17}:
            return mlp_0_3_output == 8

    attn_2_5_pattern = select_closest(mlp_0_3_outputs, positions, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, mlp_0_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_1_output, mlp_0_2_output):
        if attn_0_1_output in {"0"}:
            return mlp_0_2_output == 18
        elif attn_0_1_output in {"4", "3", "1"}:
            return mlp_0_2_output == 2
        elif attn_0_1_output in {"2"}:
            return mlp_0_2_output == 4
        elif attn_0_1_output in {"5"}:
            return mlp_0_2_output == 13
        elif attn_0_1_output in {"<s>"}:
            return mlp_0_2_output == 0

    attn_2_6_pattern = select_closest(mlp_0_2_outputs, attn_0_1_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_0_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_1_output, mlp_0_2_output):
        if attn_0_1_output in {"<s>", "0"}:
            return mlp_0_2_output == 18
        elif attn_0_1_output in {"3", "2", "1"}:
            return mlp_0_2_output == 17
        elif attn_0_1_output in {"4"}:
            return mlp_0_2_output == 2
        elif attn_0_1_output in {"5"}:
            return mlp_0_2_output == 15

    attn_2_7_pattern = select_closest(mlp_0_2_outputs, attn_0_1_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_5_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(token, attn_0_7_output):
        if token in {"4", "<s>", "0", "5", "1"}:
            return attn_0_7_output == ""
        elif token in {"2"}:
            return attn_0_7_output == "<pad>"
        elif token in {"3"}:
            return attn_0_7_output == "3"

    num_attn_2_0_pattern = select(attn_0_7_outputs, tokens, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_1_output, mlp_0_1_output):
        if attn_0_1_output in {"0"}:
            return mlp_0_1_output == 2
        elif attn_0_1_output in {"4", "3", "2", "1"}:
            return mlp_0_1_output == 18
        elif attn_0_1_output in {"5"}:
            return mlp_0_1_output == 17
        elif attn_0_1_output in {"<s>"}:
            return mlp_0_1_output == 19

    num_attn_2_1_pattern = select(mlp_0_1_outputs, attn_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_6_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_1_output, attn_0_3_output):
        if attn_0_1_output in {"4", "0", "3", "5", "1"}:
            return attn_0_3_output == ""
        elif attn_0_1_output in {"<s>", "2"}:
            return attn_0_3_output == "2"

    num_attn_2_2_pattern = select(attn_0_3_outputs, attn_0_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_7_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_0_output, attn_0_3_output):
        if attn_0_0_output in {"0", "3", "5"}:
            return attn_0_3_output == "2"
        elif attn_0_0_output in {"4", "<s>", "1"}:
            return attn_0_3_output == ""
        elif attn_0_0_output in {"2"}:
            return attn_0_3_output == "5"

    num_attn_2_3_pattern = select(attn_0_3_outputs, attn_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(q_mlp_0_3_output, k_mlp_0_3_output):
        if q_mlp_0_3_output in {0, 2}:
            return k_mlp_0_3_output == 17
        elif q_mlp_0_3_output in {1, 11}:
            return k_mlp_0_3_output == 0
        elif q_mlp_0_3_output in {3}:
            return k_mlp_0_3_output == 3
        elif q_mlp_0_3_output in {8, 10, 4}:
            return k_mlp_0_3_output == 2
        elif q_mlp_0_3_output in {5, 7, 9, 15, 16, 17}:
            return k_mlp_0_3_output == 18
        elif q_mlp_0_3_output in {6}:
            return k_mlp_0_3_output == 1
        elif q_mlp_0_3_output in {18, 12, 13, 14}:
            return k_mlp_0_3_output == 16
        elif q_mlp_0_3_output in {19}:
            return k_mlp_0_3_output == 14

    num_attn_2_4_pattern = select(mlp_0_3_outputs, mlp_0_3_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_4_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(token, attn_0_4_output):
        if token in {"0", "3", "5", "1", "2"}:
            return attn_0_4_output == "4"
        elif token in {"4", "<s>"}:
            return attn_0_4_output == ""

    num_attn_2_5_pattern = select(attn_0_4_outputs, tokens, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_3_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_1_output, token):
        if attn_0_1_output in {"0", "5"}:
            return token == ""
        elif attn_0_1_output in {"2", "1"}:
            return token == "4"
        elif attn_0_1_output in {"<s>", "3"}:
            return token == "3"
        elif attn_0_1_output in {"4"}:
            return token == "5"

    num_attn_2_6_pattern = select(tokens, attn_0_1_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_3_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_7_output, attn_0_3_output):
        if attn_0_7_output in {"0", "3", "5"}:
            return attn_0_3_output == "2"
        elif attn_0_7_output in {"2", "1"}:
            return attn_0_3_output == "5"
        elif attn_0_7_output in {"4", "<s>"}:
            return attn_0_3_output == ""

    num_attn_2_7_pattern = select(attn_0_3_outputs, attn_0_7_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_1_1_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(position, attn_2_4_output):
        key = (position, attn_2_4_output)
        if key in {
            (0, 0),
            (0, 2),
            (0, 3),
            (0, 9),
            (0, 12),
            (0, 14),
            (0, 17),
            (0, 18),
            (0, 19),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (2, 0),
            (2, 2),
            (2, 3),
            (2, 6),
            (2, 7),
            (2, 9),
            (2, 12),
            (2, 14),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (3, 10),
            (3, 12),
            (3, 14),
            (3, 18),
            (3, 19),
            (4, 12),
            (4, 18),
            (5, 12),
            (5, 14),
            (5, 18),
            (5, 19),
            (6, 12),
            (6, 18),
            (10, 10),
            (10, 12),
            (10, 14),
            (10, 18),
            (10, 19),
            (11, 12),
            (11, 14),
            (11, 18),
            (11, 19),
            (12, 12),
            (12, 18),
            (12, 19),
            (13, 12),
            (13, 18),
            (14, 0),
            (14, 12),
            (14, 14),
            (14, 17),
            (14, 18),
            (14, 19),
            (15, 12),
            (15, 18),
            (15, 19),
            (16, 12),
            (16, 18),
            (16, 19),
            (17, 0),
            (17, 12),
            (17, 18),
            (17, 19),
            (18, 0),
            (18, 12),
            (18, 14),
            (18, 17),
            (18, 18),
            (18, 19),
            (19, 12),
            (19, 18),
        }:
            return 13
        elif key in {
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 9),
            (7, 10),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (7, 19),
            (8, 2),
            (8, 3),
            (8, 6),
            (8, 10),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 18),
            (8, 19),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 5),
            (9, 6),
            (9, 10),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 18),
            (9, 19),
            (16, 13),
            (16, 14),
        }:
            return 11
        elif key in {
            (0, 10),
            (0, 15),
            (1, 4),
            (2, 1),
            (2, 4),
            (2, 5),
            (2, 10),
            (2, 11),
            (2, 15),
            (5, 10),
            (5, 15),
            (10, 15),
            (11, 10),
            (11, 15),
            (13, 15),
            (15, 10),
            (15, 15),
            (17, 10),
            (17, 15),
            (18, 10),
            (18, 15),
            (19, 15),
        }:
            return 15
        return 10

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(positions, attn_2_4_outputs)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_5_output):
        key = attn_1_5_output
        return 9

    mlp_2_1_outputs = [mlp_2_1(k0) for k0 in attn_1_5_outputs]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(num_mlp_1_2_output, attn_0_3_output):
        key = (num_mlp_1_2_output, attn_0_3_output)
        return 6

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(num_mlp_1_2_outputs, attn_0_3_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(mlp_1_0_output, attn_2_5_output):
        key = (mlp_1_0_output, attn_2_5_output)
        if key in {(0, 7), (2, 7), (4, 7), (6, 7), (7, 7), (11, 7), (12, 7), (18, 7)}:
            return 3
        return 10

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(mlp_1_0_outputs, attn_2_5_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_4_output, num_attn_1_0_output):
        key = (num_attn_1_4_output, num_attn_1_0_output)
        return 1

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_6_output, num_attn_2_7_output):
        key = (num_attn_1_6_output, num_attn_2_7_output)
        return 6

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_2_7_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_0_output, num_attn_2_3_output):
        key = (num_attn_2_0_output, num_attn_2_3_output)
        return 15

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_3_output, num_attn_2_0_output):
        key = (num_attn_2_3_output, num_attn_2_0_output)
        return 11

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_0_outputs)
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
