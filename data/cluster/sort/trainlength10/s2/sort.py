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
        "output/length/rasp/sort/trainlength10/s2/sort_weights.csv",
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
        if position in {0, 8, 9, 11, 19}:
            return token == "0"
        elif position in {1, 7}:
            return token == "1"
        elif position in {2, 6}:
            return token == "2"
        elif position in {3, 4, 5}:
            return token == "3"
        elif position in {10, 12, 13, 14, 15, 16, 17, 18}:
            return token == ""

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"1", "0"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"</s>", "4"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == "1"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 5, 9}:
            return token == "0"
        elif position in {8, 2, 6}:
            return token == "4"
        elif position in {3, 4}:
            return token == "3"
        elif position in {7}:
            return token == "2"
        elif position in {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 8, 3}:
            return token == "4"
        elif position in {1, 5, 9, 16, 18, 19}:
            return token == "0"
        elif position in {2, 4}:
            return token == "2"
        elif position in {6}:
            return token == "1"
        elif position in {7}:
            return token == "3"
        elif position in {10, 11, 12, 13, 14, 15, 17}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 6}:
            return k_position == 2
        elif q_position in {8, 2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 1
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10, 12, 13}:
            return k_position == 19
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {19, 14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16, 18}:
            return k_position == 18
        elif q_position in {17}:
            return k_position == 12

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 1, 2, 9}:
            return token == "0"
        elif position in {8, 3, 4, 7}:
            return token == "3"
        elif position in {5}:
            return token == "2"
        elif position in {6}:
            return token == "1"
        elif position in {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0, 9, 2}:
            return token == "2"
        elif position in {1, 7}:
            return token == "1"
        elif position in {8, 3, 4}:
            return token == "4"
        elif position in {5}:
            return token == "3"
        elif position in {6}:
            return token == "0"
        elif position in {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""
        elif position in {3}:
            return token == "3"
        elif position in {8, 4, 7}:
            return token == "4"
        elif position in {9, 5}:
            return token == "0"
        elif position in {6}:
            return token == "1"
        elif position in {10}:
            return token == "</s>"

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 9}:
            return token == "0"
        elif position in {1, 2, 3, 10, 14, 16, 17}:
            return token == ""
        elif position in {4, 11, 12, 13, 18, 19}:
            return token == "3"
        elif position in {5, 6, 15}:
            return token == "2"
        elif position in {8, 7}:
            return token == "1"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 2, 3, 13, 14, 15, 17, 19}:
            return token == "0"
        elif position in {4}:
            return token == "</s>"
        elif position in {5, 6, 11, 16, 18}:
            return token == "3"
        elif position in {8, 10, 12, 7}:
            return token == ""
        elif position in {9}:
            return token == "2"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 6}:
            return token == "0"
        elif position in {1, 2, 3, 4, 14, 15, 16, 17, 19}:
            return token == ""
        elif position in {5, 10, 11, 12, 13, 18}:
            return token == "3"
        elif position in {7}:
            return token == "2"
        elif position in {8, 9}:
            return token == "1"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 9}:
            return token == "2"
        elif position in {1}:
            return token == "</s>"
        elif position in {2}:
            return token == "1"
        elif position in {10, 3, 4, 14}:
            return token == "0"
        elif position in {5, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19}:
            return token == ""

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 9, 3, 4}:
            return token == "1"
        elif position in {1}:
            return token == "<pad>"
        elif position in {2, 10, 11, 13, 15, 16, 17}:
            return token == ""
        elif position in {8, 5, 6, 7}:
            return token == "0"
        elif position in {18, 19, 12, 14}:
            return token == "2"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 2, 3, 4, 10, 11, 12, 14, 15, 16}:
            return token == ""
        elif position in {5}:
            return token == "</s>"
        elif position in {6, 7, 9, 18, 19}:
            return token == "3"
        elif position in {8}:
            return token == "2"
        elif position in {17, 13}:
            return token == "4"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 9}:
            return token == "0"
        elif position in {8, 1, 13, 17}:
            return token == ""
        elif position in {2}:
            return token == "</s>"
        elif position in {3, 4, 10, 11, 12, 14, 15, 16, 18, 19}:
            return token == "2"
        elif position in {5, 6}:
            return token == "1"
        elif position in {7}:
            return token == "<s>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 10, 11, 14, 15, 17, 18, 19}:
            return token == "0"
        elif position in {1}:
            return token == "<s>"
        elif position in {2, 3, 4, 12, 13, 16}:
            return token == "1"
        elif position in {8, 5, 6, 7}:
            return token == ""
        elif position in {9}:
            return token == "2"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {0, 5, 6, 7, 8}:
            return 4
        elif key in {1, 2, 9, 11}:
            return 6
        return 5

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {0, 1, 9, 10, 11, 12, 16, 19}:
            return 3
        elif key in {6, 7, 8}:
            return 17
        return 14

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(position, attn_0_5_output):
        key = (position, attn_0_5_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "0"),
            (8, "0"),
            (8, "1"),
            (8, "2"),
            (8, "3"),
            (8, "4"),
            (8, "</s>"),
            (8, "<s>"),
            (9, "0"),
            (9, "1"),
            (9, "2"),
            (9, "3"),
            (9, "4"),
            (9, "</s>"),
            (9, "<s>"),
            (10, "0"),
            (10, "2"),
            (10, "3"),
            (11, "0"),
            (11, "2"),
            (11, "3"),
            (12, "0"),
            (12, "3"),
            (13, "0"),
            (13, "3"),
            (14, "0"),
            (14, "3"),
            (15, "0"),
            (15, "3"),
            (16, "0"),
            (16, "3"),
            (17, "0"),
            (17, "3"),
            (18, "0"),
            (19, "0"),
            (19, "3"),
        }:
            return 10
        return 6

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(positions, attn_0_5_outputs)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {
            ("0", 0),
            ("0", 2),
            ("0", 3),
            ("0", 9),
            ("1", 2),
            ("1", 3),
            ("1", 9),
            ("2", 0),
            ("2", 2),
            ("2", 3),
            ("2", 9),
            ("2", 19),
            ("3", 0),
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("3", 9),
            ("3", 10),
            ("3", 11),
            ("3", 14),
            ("3", 15),
            ("3", 16),
            ("3", 19),
            ("4", 0),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 14),
            ("4", 15),
            ("4", 16),
            ("4", 17),
            ("4", 19),
            ("</s>", 0),
            ("</s>", 1),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 9),
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 9),
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
            return 11
        elif key in {
            ("0", 1),
            ("1", 1),
            ("1", 10),
            ("1", 11),
            ("1", 16),
            ("1", 19),
            ("2", 1),
        }:
            return 8
        return 2

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_7_output, num_attn_0_1_output):
        key = (num_attn_0_7_output, num_attn_0_1_output)
        return 16

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_0_output):
        key = (num_attn_0_3_output, num_attn_0_0_output)
        return 12

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_7_output, num_attn_0_4_output):
        key = (num_attn_0_7_output, num_attn_0_4_output)
        return 12

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_3_output, num_attn_0_2_output):
        key = (num_attn_0_3_output, num_attn_0_2_output)
        return 12

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 17}:
            return k_position == 5
        elif q_position in {1, 4, 9}:
            return k_position == 3
        elif q_position in {2, 3, 10, 11, 19}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6, 14}:
            return k_position == 8
        elif q_position in {18, 13, 7}:
            return k_position == 6
        elif q_position in {8, 16}:
            return k_position == 7
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {15}:
            return k_position == 9

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_1_output, attn_0_7_output):
        if mlp_0_1_output in {0, 2, 9, 10, 15, 19}:
            return attn_0_7_output == "</s>"
        elif mlp_0_1_output in {1, 5, 7, 16, 18}:
            return attn_0_7_output == ""
        elif mlp_0_1_output in {3, 13}:
            return attn_0_7_output == "1"
        elif mlp_0_1_output in {4}:
            return attn_0_7_output == "2"
        elif mlp_0_1_output in {8, 11, 6}:
            return attn_0_7_output == "0"
        elif mlp_0_1_output in {12}:
            return attn_0_7_output == "<pad>"
        elif mlp_0_1_output in {14}:
            return attn_0_7_output == "<s>"
        elif mlp_0_1_output in {17}:
            return attn_0_7_output == "3"

    attn_1_1_pattern = select_closest(attn_0_7_outputs, mlp_0_1_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_4_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, token):
        if position in {0, 5}:
            return token == "1"
        elif position in {1, 3, 6, 7, 8, 11}:
            return token == "0"
        elif position in {2}:
            return token == "2"
        elif position in {4, 13, 16, 18, 19}:
            return token == "</s>"
        elif position in {9}:
            return token == "4"
        elif position in {10, 12, 14, 15, 17}:
            return token == ""

    attn_1_2_pattern = select_closest(tokens, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 2, 9}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {18, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 1
        elif q_position in {10, 13, 14}:
            return k_position == 19
        elif q_position in {11}:
            return k_position == 11
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16, 17}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 17

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, mlp_0_3_output):
        if position in {0, 14, 7}:
            return mlp_0_3_output == 4
        elif position in {1, 9}:
            return mlp_0_3_output == 2
        elif position in {2, 3, 4, 5, 6, 10, 12, 15, 19}:
            return mlp_0_3_output == 8
        elif position in {8, 18, 13}:
            return mlp_0_3_output == 5
        elif position in {11}:
            return mlp_0_3_output == 14
        elif position in {16}:
            return mlp_0_3_output == 6
        elif position in {17}:
            return mlp_0_3_output == 3

    attn_1_4_pattern = select_closest(mlp_0_3_outputs, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1, 9}:
            return token == "1"
        elif position in {2, 11}:
            return token == "0"
        elif position in {19, 3, 14, 15}:
            return token == "2"
        elif position in {4}:
            return token == "</s>"
        elif position in {8, 5, 6, 7}:
            return token == "4"
        elif position in {10, 12, 13, 16, 17, 18}:
            return token == ""

    attn_1_5_pattern = select_closest(tokens, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, mlp_0_0_output):
        if position in {0, 9}:
            return mlp_0_0_output == 4
        elif position in {1, 12, 13, 15, 19}:
            return mlp_0_0_output == 3
        elif position in {2}:
            return mlp_0_0_output == 7
        elif position in {3, 4}:
            return mlp_0_0_output == 6
        elif position in {8, 5, 6, 7}:
            return mlp_0_0_output == 5
        elif position in {10}:
            return mlp_0_0_output == 15
        elif position in {11}:
            return mlp_0_0_output == 1
        elif position in {16, 14}:
            return mlp_0_0_output == 9
        elif position in {17}:
            return mlp_0_0_output == 14
        elif position in {18}:
            return mlp_0_0_output == 10

    attn_1_6_pattern = select_closest(mlp_0_0_outputs, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(mlp_0_1_output, attn_0_0_output):
        if mlp_0_1_output in {0, 4, 5, 6, 10, 13, 15, 16, 19}:
            return attn_0_0_output == "</s>"
        elif mlp_0_1_output in {8, 1, 14}:
            return attn_0_0_output == ""
        elif mlp_0_1_output in {2}:
            return attn_0_0_output == "2"
        elif mlp_0_1_output in {3, 12}:
            return attn_0_0_output == "1"
        elif mlp_0_1_output in {7}:
            return attn_0_0_output == "<pad>"
        elif mlp_0_1_output in {9, 17}:
            return attn_0_0_output == "3"
        elif mlp_0_1_output in {11}:
            return attn_0_0_output == "0"
        elif mlp_0_1_output in {18}:
            return attn_0_0_output == "4"

    attn_1_7_pattern = select_closest(attn_0_0_outputs, mlp_0_1_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_5_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19}:
            return k_mlp_0_1_output == 17
        elif q_mlp_0_1_output in {4, 5, 6, 7}:
            return k_mlp_0_1_output == 3
        elif q_mlp_0_1_output in {8}:
            return k_mlp_0_1_output == 4
        elif q_mlp_0_1_output in {17}:
            return k_mlp_0_1_output == 0

    num_attn_1_0_pattern = select(mlp_0_1_outputs, mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, attn_0_0_output):
        if position in {0, 9, 19}:
            return attn_0_0_output == "2"
        elif position in {1, 2, 3, 4, 11, 12, 15, 18}:
            return attn_0_0_output == "0"
        elif position in {5, 10, 13, 14, 16}:
            return attn_0_0_output == "1"
        elif position in {6}:
            return attn_0_0_output == "</s>"
        elif position in {8, 17, 7}:
            return attn_0_0_output == ""

    num_attn_1_1_pattern = select(attn_0_0_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"3", "1", "4", "2", "</s>", "<s>", "0"}:
            return k_attn_0_6_output == "2"

    num_attn_1_2_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_0_output):
        if position in {0, 6, 10, 17, 18, 19}:
            return attn_0_0_output == "1"
        elif position in {1, 9}:
            return attn_0_0_output == "3"
        elif position in {2, 3, 4, 5, 11, 12, 13, 14, 15, 16}:
            return attn_0_0_output == "2"
        elif position in {8, 7}:
            return attn_0_0_output == ""

    num_attn_1_3_pattern = select(attn_0_0_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(mlp_0_3_output, attn_0_6_output):
        if mlp_0_3_output in {0, 1, 10, 12}:
            return attn_0_6_output == "1"
        elif mlp_0_3_output in {2}:
            return attn_0_6_output == "<s>"
        elif mlp_0_3_output in {3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 18, 19}:
            return attn_0_6_output == "0"
        elif mlp_0_3_output in {17}:
            return attn_0_6_output == ""

    num_attn_1_4_pattern = select(attn_0_6_outputs, mlp_0_3_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_4_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(num_mlp_0_3_output, attn_0_3_output):
        if num_mlp_0_3_output in {0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 16, 17, 19}:
            return attn_0_3_output == "2"
        elif num_mlp_0_3_output in {18, 5, 6, 15}:
            return attn_0_3_output == "0"
        elif num_mlp_0_3_output in {9}:
            return attn_0_3_output == "</s>"

    num_attn_1_5_pattern = select(
        attn_0_3_outputs, num_mlp_0_3_outputs, num_predicate_1_5
    )
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_3_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(num_mlp_0_2_output, token):
        if num_mlp_0_2_output in {0, 2, 3, 9, 10, 13, 17, 18, 19}:
            return token == "2"
        elif num_mlp_0_2_output in {1, 5, 8, 11, 12, 15, 16}:
            return token == "<s>"
        elif num_mlp_0_2_output in {4}:
            return token == "3"
        elif num_mlp_0_2_output in {6}:
            return token == "1"
        elif num_mlp_0_2_output in {7}:
            return token == "0"
        elif num_mlp_0_2_output in {14}:
            return token == "</s>"

    num_attn_1_6_pattern = select(tokens, num_mlp_0_2_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_3_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, attn_0_0_output):
        if position in {0, 9}:
            return attn_0_0_output == "2"
        elif position in {1, 3, 4, 10, 11, 12, 13, 14, 15, 16}:
            return attn_0_0_output == "1"
        elif position in {2, 5, 6, 7, 17, 18, 19}:
            return attn_0_0_output == "0"
        elif position in {8}:
            return attn_0_0_output == ""

    num_attn_1_7_pattern = select(attn_0_0_outputs, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_0_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_5_output):
        key = (position, attn_1_5_output)
        if key in {
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "</s>"),
            (0, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "</s>"),
            (4, "<s>"),
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "</s>"),
            (5, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "</s>"),
            (6, "<s>"),
            (9, "2"),
            (9, "3"),
            (9, "4"),
            (9, "</s>"),
            (9, "<s>"),
            (10, "2"),
            (10, "3"),
            (10, "4"),
            (10, "</s>"),
            (10, "<s>"),
            (12, "2"),
            (12, "4"),
            (12, "<s>"),
            (13, "2"),
            (13, "4"),
            (13, "<s>"),
            (14, "2"),
            (14, "3"),
            (14, "4"),
            (14, "</s>"),
            (14, "<s>"),
            (15, "2"),
            (15, "3"),
            (15, "4"),
            (15, "</s>"),
            (15, "<s>"),
            (16, "2"),
            (16, "3"),
            (16, "4"),
            (16, "</s>"),
            (16, "<s>"),
            (17, "1"),
            (17, "2"),
            (17, "3"),
            (17, "4"),
            (17, "</s>"),
            (17, "<s>"),
            (18, "2"),
            (18, "3"),
            (18, "4"),
            (18, "</s>"),
            (18, "<s>"),
            (19, "2"),
            (19, "3"),
            (19, "4"),
            (19, "</s>"),
            (19, "<s>"),
        }:
            return 9
        elif key in {(1, "0")}:
            return 4
        elif key in {(0, "0"), (1, "2"), (1, "4"), (1, "<s>"), (11, "<s>")}:
            return 8
        elif key in {(1, "1"), (1, "</s>")}:
            return 1
        return 11

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_5_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_1_0_output):
        key = (position, attn_1_0_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (10, "0"),
            (10, "2"),
            (10, "3"),
            (11, "0"),
            (11, "2"),
            (11, "3"),
            (12, "0"),
            (12, "2"),
            (12, "3"),
            (13, "0"),
            (13, "2"),
            (13, "3"),
            (14, "0"),
            (14, "2"),
            (14, "3"),
            (15, "0"),
            (15, "2"),
            (15, "3"),
            (16, "0"),
            (16, "2"),
            (16, "3"),
            (17, "0"),
            (17, "2"),
            (17, "3"),
            (18, "0"),
            (18, "2"),
            (18, "3"),
            (19, "0"),
            (19, "2"),
            (19, "3"),
        }:
            return 0
        return 11

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_1_0_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_1_output, position):
        key = (attn_1_1_output, position)
        if key in {
            ("0", 1),
            ("1", 0),
            ("1", 1),
            ("1", 6),
            ("1", 7),
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
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 16
        elif key in {("</s>", 7)}:
            return 17
        return 13

    mlp_1_2_outputs = [mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_1_outputs, positions)]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_5_output, position):
        key = (attn_1_5_output, position)
        if key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("0", 15),
            ("0", 16),
            ("0", 18),
            ("0", 19),
            ("1", 2),
            ("1", 9),
            ("2", 1),
            ("2", 2),
            ("2", 9),
            ("3", 2),
            ("3", 9),
            ("4", 2),
            ("4", 9),
            ("</s>", 1),
            ("</s>", 2),
            ("</s>", 9),
        }:
            return 7
        elif key in {
            ("0", 8),
            ("1", 8),
            ("2", 8),
            ("3", 8),
            ("4", 8),
            ("</s>", 8),
            ("<s>", 8),
        }:
            return 1
        elif key in {("1", 1), ("4", 1)}:
            return 14
        elif key in {
            ("1", 0),
            ("1", 7),
            ("1", 10),
            ("1", 11),
            ("1", 12),
            ("1", 13),
            ("1", 15),
            ("1", 16),
            ("1", 18),
            ("1", 19),
            ("<s>", 2),
            ("<s>", 9),
        }:
            return 16
        return 9

    mlp_1_3_outputs = [mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_5_outputs, positions)]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_4_output, num_attn_1_0_output):
        key = (num_attn_1_4_output, num_attn_1_0_output)
        return 19

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_7_output, num_attn_0_3_output):
        key = (num_attn_0_7_output, num_attn_0_3_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (3, 0),
        }:
            return 11
        return 16

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_0_3_output, num_attn_0_7_output):
        key = (num_attn_0_3_output, num_attn_0_7_output)
        if key in {(0, 0), (0, 1), (1, 0)}:
            return 17
        return 12

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_4_output, num_attn_1_3_output):
        key = (num_attn_1_4_output, num_attn_1_3_output)
        return 6

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, token):
        if position in {0, 2, 3, 12, 13, 16, 18}:
            return token == ""
        elif position in {1}:
            return token == "2"
        elif position in {4}:
            return token == "<s>"
        elif position in {5, 6, 7, 9, 10, 11, 14, 15, 17, 19}:
            return token == "</s>"
        elif position in {8}:
            return token == "4"

    attn_2_0_pattern = select_closest(tokens, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_5_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_3_output, mlp_1_1_output):
        if attn_1_3_output in {"0"}:
            return mlp_1_1_output == 2
        elif attn_1_3_output in {"1", "3", "2", "</s>"}:
            return mlp_1_1_output == 0
        elif attn_1_3_output in {"4", "<s>"}:
            return mlp_1_1_output == 11

    attn_2_1_pattern = select_closest(mlp_1_1_outputs, attn_1_3_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_7_output, token):
        if attn_0_7_output in {"1", "0"}:
            return token == "</s>"
        elif attn_0_7_output in {"2", "</s>"}:
            return token == "<s>"
        elif attn_0_7_output in {"3"}:
            return token == "1"
        elif attn_0_7_output in {"4"}:
            return token == ""
        elif attn_0_7_output in {"<s>"}:
            return token == "<pad>"

    attn_2_2_pattern = select_closest(tokens, attn_0_7_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(position, token):
        if position in {0, 17, 19, 15}:
            return token == ""
        elif position in {1}:
            return token == "0"
        elif position in {2, 3, 7, 8, 12, 18}:
            return token == "3"
        elif position in {4, 5}:
            return token == "4"
        elif position in {6}:
            return token == "</s>"
        elif position in {9, 10, 11, 13, 14, 16}:
            return token == "<s>"

    attn_2_3_pattern = select_closest(tokens, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, tokens)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(num_mlp_1_2_output, token):
        if num_mlp_1_2_output in {0, 1, 5, 8, 12, 13, 18}:
            return token == ""
        elif num_mlp_1_2_output in {2, 7, 10, 11, 14, 15, 17}:
            return token == "<s>"
        elif num_mlp_1_2_output in {3, 4, 6, 9, 16}:
            return token == "</s>"
        elif num_mlp_1_2_output in {19}:
            return token == "1"

    attn_2_4_pattern = select_closest(tokens, num_mlp_1_2_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_0_4_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(position, token):
        if position in {0, 10, 4, 15}:
            return token == "<s>"
        elif position in {1}:
            return token == "2"
        elif position in {9, 2, 14}:
            return token == "0"
        elif position in {3}:
            return token == "<pad>"
        elif position in {8, 13, 5}:
            return token == "3"
        elif position in {6}:
            return token == "</s>"
        elif position in {7, 11, 12, 16, 17, 18}:
            return token == ""
        elif position in {19}:
            return token == "1"

    attn_2_5_pattern = select_closest(tokens, positions, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_3_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(position, token):
        if position in {0, 17, 13, 14}:
            return token == ""
        elif position in {1, 2, 12}:
            return token == "0"
        elif position in {3, 4, 6, 9, 18, 19}:
            return token == "</s>"
        elif position in {8, 5, 7}:
            return token == "4"
        elif position in {10}:
            return token == "3"
        elif position in {16, 11, 15}:
            return token == "<s>"

    attn_2_6_pattern = select_closest(tokens, positions, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, tokens)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 11, 13}:
            return token == "0"
        elif mlp_0_1_output in {1}:
            return token == "<pad>"
        elif mlp_0_1_output in {8, 16, 2, 18}:
            return token == "2"
        elif mlp_0_1_output in {3, 6}:
            return token == "3"
        elif mlp_0_1_output in {4, 5, 7, 9, 10, 15, 19}:
            return token == ""
        elif mlp_0_1_output in {17, 12, 14}:
            return token == "4"

    attn_2_7_pattern = select_closest(tokens, mlp_0_1_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, tokens)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 2, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16, 18, 19}:
            return token == "0"
        elif mlp_0_1_output in {1, 10}:
            return token == "1"
        elif mlp_0_1_output in {8, 17, 6}:
            return token == ""
        elif mlp_0_1_output in {9}:
            return token == "2"

    num_attn_2_0_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_0_1_output, attn_0_4_output):
        if mlp_0_1_output in {0, 15}:
            return attn_0_4_output == "0"
        elif mlp_0_1_output in {1, 3, 4, 5, 10, 11, 12, 13, 14, 16, 19}:
            return attn_0_4_output == "1"
        elif mlp_0_1_output in {9, 2, 18, 6}:
            return attn_0_4_output == "2"
        elif mlp_0_1_output in {7}:
            return attn_0_4_output == "<s>"
        elif mlp_0_1_output in {8, 17}:
            return attn_0_4_output == ""

    num_attn_2_1_pattern = select(attn_0_4_outputs, mlp_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_4_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, token):
        if position in {0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 17, 18, 19}:
            return token == "2"
        elif position in {8, 1, 14, 16}:
            return token == ""
        elif position in {7}:
            return token == "</s>"

    num_attn_2_2_pattern = select(tokens, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_6_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_7_output, token):
        if attn_1_7_output in {"4", "0"}:
            return token == "<s>"
        elif attn_1_7_output in {"1", "3", "</s>"}:
            return token == "1"
        elif attn_1_7_output in {"2"}:
            return token == "2"
        elif attn_1_7_output in {"<s>"}:
            return token == "</s>"

    num_attn_2_3_pattern = select(tokens, attn_1_7_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_4_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(q_attn_1_7_output, k_attn_1_7_output):
        if q_attn_1_7_output in {"1", "2", "</s>", "<s>", "0"}:
            return k_attn_1_7_output == "3"
        elif q_attn_1_7_output in {"3"}:
            return k_attn_1_7_output == "2"
        elif q_attn_1_7_output in {"4"}:
            return k_attn_1_7_output == "4"

    num_attn_2_4_pattern = select(attn_1_7_outputs, attn_1_7_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_5_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_1_output, attn_0_4_output):
        if attn_1_1_output in {"3", "1", "2", "</s>", "<s>", "0"}:
            return attn_0_4_output == "0"
        elif attn_1_1_output in {"4"}:
            return attn_0_4_output == "3"

    num_attn_2_5_pattern = select(attn_0_4_outputs, attn_1_1_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_4_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_7_output, attn_1_0_output):
        if attn_1_7_output in {"3", "4", "2", "</s>", "<s>", "0"}:
            return attn_1_0_output == "1"
        elif attn_1_7_output in {"1"}:
            return attn_1_0_output == "2"

    num_attn_2_6_pattern = select(attn_1_0_outputs, attn_1_7_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_0_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(position, attn_1_2_output):
        if position in {0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19}:
            return attn_1_2_output == "0"
        elif position in {17, 7}:
            return attn_1_2_output == ""
        elif position in {8}:
            return attn_1_2_output == "<pad>"

    num_attn_2_7_pattern = select(attn_1_2_outputs, positions, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_4_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(position, attn_2_2_output):
        key = (position, attn_2_2_output)
        if key in {
            (0, "1"),
            (0, "2"),
            (0, "4"),
            (0, "<s>"),
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "</s>"),
            (3, "<s>"),
            (8, "0"),
            (8, "1"),
            (8, "2"),
            (8, "4"),
            (8, "</s>"),
            (8, "<s>"),
            (9, "1"),
            (9, "2"),
            (9, "4"),
            (9, "</s>"),
            (9, "<s>"),
            (10, "1"),
            (10, "2"),
            (10, "4"),
            (10, "</s>"),
            (10, "<s>"),
            (11, "1"),
            (11, "2"),
            (11, "4"),
            (11, "<s>"),
            (12, "1"),
            (12, "2"),
            (12, "4"),
            (12, "</s>"),
            (12, "<s>"),
            (13, "1"),
            (13, "2"),
            (13, "4"),
            (13, "</s>"),
            (13, "<s>"),
            (14, "1"),
            (14, "2"),
            (14, "4"),
            (14, "<s>"),
            (15, "1"),
            (15, "2"),
            (15, "4"),
            (15, "</s>"),
            (15, "<s>"),
            (16, "1"),
            (16, "2"),
            (16, "4"),
            (16, "</s>"),
            (16, "<s>"),
            (17, "4"),
            (18, "1"),
            (18, "2"),
            (18, "4"),
            (18, "</s>"),
            (18, "<s>"),
            (19, "1"),
            (19, "2"),
            (19, "4"),
            (19, "</s>"),
            (19, "<s>"),
        }:
            return 18
        return 9

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(positions, attn_2_2_outputs)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_2_output, attn_2_1_output):
        key = (attn_2_2_output, attn_2_1_output)
        if key in {
            ("0", "2"),
            ("2", "0"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "</s>"),
            ("2", "<s>"),
        }:
            return 6
        elif key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "3"),
            ("0", "4"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "3"),
            ("1", "4"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("4", "0"),
            ("4", "1"),
            ("4", "3"),
            ("4", "4"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 19
        return 3

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_1_6_output, num_mlp_1_1_output):
        key = (attn_1_6_output, num_mlp_1_1_output)
        if key in {
            ("1", 0),
            ("2", 0),
            ("2", 2),
            ("2", 16),
            ("2", 19),
            ("3", 0),
            ("3", 16),
            ("4", 0),
            ("4", 16),
            ("</s>", 0),
            ("</s>", 2),
            ("</s>", 16),
            ("</s>", 19),
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 16),
            ("<s>", 19),
        }:
            return 15
        return 16

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_1_6_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_2_1_output, position):
        key = (attn_2_1_output, position)
        if key in {
            ("0", 8),
            ("1", 8),
            ("2", 0),
            ("2", 1),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("2", 7),
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
            ("3", 3),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 17),
        }:
            return 9
        elif key in {
            ("2", 8),
            ("3", 0),
            ("3", 2),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 8),
            ("3", 9),
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
            ("4", 0),
            ("4", 2),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 8),
            ("4", 9),
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
            ("</s>", 0),
            ("</s>", 2),
            ("</s>", 4),
            ("</s>", 5),
            ("</s>", 6),
            ("</s>", 7),
            ("</s>", 8),
            ("</s>", 9),
            ("</s>", 10),
            ("</s>", 11),
            ("</s>", 12),
            ("</s>", 13),
            ("</s>", 14),
            ("</s>", 15),
            ("</s>", 16),
            ("</s>", 17),
            ("</s>", 18),
            ("</s>", 19),
            ("<s>", 8),
        }:
            return 6
        elif key in {("3", 1), ("4", 1), ("</s>", 1)}:
            return 7
        return 2

    mlp_2_3_outputs = [mlp_2_3(k0, k1) for k0, k1 in zip(attn_2_1_outputs, positions)]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_2_3_output):
        key = (num_attn_1_2_output, num_attn_2_3_output)
        return 14

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_3_output, num_attn_0_3_output):
        key = (num_attn_1_3_output, num_attn_0_3_output)
        if key in {(0, 0)}:
            return 19
        return 13

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_5_output, num_attn_2_4_output):
        key = (num_attn_1_5_output, num_attn_2_4_output)
        return 12

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_2_4_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0, 1}:
            return 2
        return 10

    num_mlp_2_3_outputs = [num_mlp_2_3(k0) for k0 in num_attn_0_3_outputs]
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


print(run(["<s>", "0", "</s>"]))
