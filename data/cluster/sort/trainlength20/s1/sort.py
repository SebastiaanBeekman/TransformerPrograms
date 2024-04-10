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
        "output/length/rasp/sort/trainlength20/s1/sort_weights.csv",
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
        if position in {0, 21, 6, 7}:
            return token == "4"
        elif position in {1, 2, 3, 4, 5, 10, 11, 14}:
            return token == "2"
        elif position in {8, 16, 19, 15}:
            return token == "0"
        elif position in {9, 17}:
            return token == "3"
        elif position in {18, 12, 13}:
            return token == "1"
        elif position in {20, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 19, 3}:
            return token == "1"
        elif position in {1, 7, 11, 14, 22}:
            return token == "2"
        elif position in {2, 4, 5, 6, 8, 9, 10, 12, 13, 16, 18}:
            return token == "4"
        elif position in {15}:
            return token == "3"
        elif position in {17}:
            return token == "0"
        elif position in {20, 21, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 3}:
            return token == "1"
        elif position in {1, 2}:
            return token == "0"
        elif position in {10, 4, 13}:
            return token == "3"
        elif position in {5, 6, 7, 8, 12, 14, 15, 17, 18}:
            return token == "4"
        elif position in {16, 9, 19}:
            return token == "2"
        elif position in {11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"0", "3"}:
            return k_token == "3"
        elif q_token in {"</s>", "4", "1"}:
            return k_token == "4"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_token, k_token):
        if q_token in {"<s>", "3", "1", "2", "0"}:
            return k_token == "3"
        elif q_token in {"</s>", "4"}:
            return k_token == "4"

    attn_0_4_pattern = select_closest(tokens, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 3, 6, 7, 11, 12, 13, 16}:
            return token == "3"
        elif position in {8, 1, 14}:
            return token == "1"
        elif position in {2, 5, 9, 15, 17}:
            return token == "4"
        elif position in {10, 4}:
            return token == "2"
        elif position in {18}:
            return token == "<s>"
        elif position in {19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 24, 12, 13}:
            return k_position == 7
        elif q_position in {1, 14}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3, 7}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {9, 29, 6}:
            return k_position == 11
        elif q_position in {8, 18, 11}:
            return k_position == 10
        elif q_position in {10, 28}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 13
        elif q_position in {16}:
            return k_position == 8
        elif q_position in {17}:
            return k_position == 9
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {25, 27, 20}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {23}:
            return k_position == 25
        elif q_position in {26}:
            return k_position == 16

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {17, 2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 29}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 24}:
            return k_position == 9
        elif q_position in {9, 23}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11, 28}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {27, 13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {16, 20}:
            return k_position == 17
        elif q_position in {18, 26}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 29

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 6, 20, 25, 26}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3, 4, 21, 23, 29}:
            return k_position == 9
        elif q_position in {24, 27, 28, 5}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {8}:
            return k_position == 14
        elif q_position in {9}:
            return k_position == 15
        elif q_position in {10}:
            return k_position == 16
        elif q_position in {11}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 18
        elif q_position in {13, 15}:
            return k_position == 21
        elif q_position in {14}:
            return k_position == 22
        elif q_position in {16, 17}:
            return k_position == 24
        elif q_position in {18}:
            return k_position == 28
        elif q_position in {19, 22}:
            return k_position == 8

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 15}:
            return token == "0"
        elif position in {1, 2, 3, 5, 21, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {4, 22}:
            return token == "<pad>"
        elif position in {9, 6, 7}:
            return token == "<s>"
        elif position in {8}:
            return token == "</s>"
        elif position in {10, 11, 14, 16, 19}:
            return token == "2"
        elif position in {17, 18, 12, 13}:
            return token == "1"
        elif position in {20}:
            return token == "3"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 9, 15}:
            return token == "1"
        elif position in {1, 2, 3, 4, 21, 22, 26, 29}:
            return token == ""
        elif position in {19, 5}:
            return token == "<s>"
        elif position in {6, 7}:
            return token == "</s>"
        elif position in {8, 24, 25}:
            return token == "2"
        elif position in {10, 11, 12, 13, 14, 16, 17, 18}:
            return token == "0"
        elif position in {20, 28}:
            return token == "<pad>"
        elif position in {27, 23}:
            return token == "3"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 25, 10, 27}:
            return k_position == 12
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9, 26}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12, 21}:
            return k_position == 15
        elif q_position in {13, 22}:
            return k_position == 16
        elif q_position in {19, 14}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16, 17, 28}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 22
        elif q_position in {24, 20, 29, 23}:
            return k_position == 1

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 8, 19, 9}:
            return token == "0"
        elif position in {1}:
            return token == "</s>"
        elif position in {2, 3, 4, 5, 6, 7, 21, 22, 23}:
            return token == "1"
        elif position in {10, 11, 12, 13, 14, 15, 16, 17, 18}:
            return token == ""
        elif position in {20, 24, 25, 26, 27, 28, 29}:
            return token == "2"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 1, 9, 13, 24, 27}:
            return token == "1"
        elif position in {19, 2, 3, 7}:
            return token == "0"
        elif position in {4, 14}:
            return token == "2"
        elif position in {5, 6, 8, 10, 11, 16, 17, 18, 20, 21, 22, 23, 25, 26, 28, 29}:
            return token == ""
        elif position in {12, 15}:
            return token == "<s>"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0, 9}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {25, 10, 20}:
            return k_position == 14
        elif q_position in {26, 11, 12}:
            return k_position == 16
        elif q_position in {13, 14}:
            return k_position == 18
        elif q_position in {15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 24
        elif q_position in {17}:
            return k_position == 26
        elif q_position in {18}:
            return k_position == 20
        elif q_position in {19}:
            return k_position == 2
        elif q_position in {21, 23, 24, 27, 28, 29}:
            return k_position == 1
        elif q_position in {22}:
            return k_position == 11

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 28}:
            return token == "1"
        elif position in {1, 2, 3, 4, 5, 19, 23, 24, 25, 26, 27, 29}:
            return token == "0"
        elif position in {6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22}:
            return token == ""
        elif position in {15}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position, attn_0_5_output):
        key = (position, attn_0_5_output)
        if key in {
            (0, "0"),
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "</s>"),
            (3, "<s>"),
            (4, "0"),
            (5, "0"),
            (5, "1"),
            (5, "3"),
            (5, "4"),
            (5, "</s>"),
            (5, "<s>"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "</s>"),
            (7, "<s>"),
            (18, "0"),
            (18, "1"),
            (18, "4"),
            (18, "</s>"),
            (19, "0"),
            (19, "1"),
            (19, "3"),
            (19, "4"),
            (19, "</s>"),
            (19, "<s>"),
            (20, "0"),
            (21, "0"),
            (22, "0"),
            (23, "0"),
            (23, "1"),
            (23, "</s>"),
            (24, "0"),
            (24, "1"),
            (24, "</s>"),
            (25, "0"),
            (26, "0"),
            (27, "0"),
            (27, "1"),
            (27, "</s>"),
            (28, "0"),
            (29, "0"),
        }:
            return 9
        elif key in {
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "</s>"),
            (0, "<s>"),
            (8, "0"),
            (8, "1"),
            (8, "2"),
            (8, "3"),
            (8, "4"),
            (8, "</s>"),
            (8, "<s>"),
            (20, "1"),
            (20, "</s>"),
            (20, "<s>"),
            (21, "1"),
            (21, "</s>"),
            (22, "1"),
            (22, "</s>"),
            (25, "1"),
            (25, "</s>"),
            (26, "1"),
            (26, "</s>"),
            (26, "<s>"),
            (28, "1"),
            (28, "</s>"),
            (29, "1"),
            (29, "</s>"),
            (29, "<s>"),
        }:
            return 23
        elif key in {
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "</s>"),
            (4, "<s>"),
            (5, "2"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "</s>"),
            (6, "<s>"),
        }:
            return 25
        elif key in {
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (19, "2"),
        }:
            return 16
        return 29

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(positions, attn_0_5_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {0, 3, 7, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return 2
        elif key in {1, 19}:
            return 27
        elif key in {2}:
            return 21
        return 16

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(position):
        key = position
        if key in {1, 2, 3, 4, 5, 20, 21, 22, 24, 25, 26, 27}:
            return 5
        elif key in {6, 19}:
            return 16
        return 7

    mlp_0_2_outputs = [mlp_0_2(k0) for k0 in positions]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position):
        key = position
        if key in {5, 12, 13, 14, 15, 16, 17, 18}:
            return 28
        elif key in {0, 4, 6, 7}:
            return 6
        elif key in {8, 9, 10, 11}:
            return 22
        elif key in {1, 2, 19, 24}:
            return 26
        elif key in {3}:
            return 23
        return 2

    mlp_0_3_outputs = [mlp_0_3(k0) for k0 in positions]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_7_output):
        key = (num_attn_0_0_output, num_attn_0_7_output)
        return 25

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 0

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 11

    num_mlp_0_2_outputs = [num_mlp_0_2(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 7

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 11}:
            return token == "3"
        elif position in {1, 19, 12, 13}:
            return token == "1"
        elif position in {2, 23}:
            return token == "</s>"
        elif position in {3, 4, 5, 6, 7, 9, 15, 16, 20, 21, 22, 24, 28, 29}:
            return token == "4"
        elif position in {8, 25, 10}:
            return token == "2"
        elif position in {17, 26, 14}:
            return token == "0"
        elif position in {18}:
            return token == "<s>"
        elif position in {27}:
            return token == ""

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_3_output, mlp_0_2_output):
        if mlp_0_3_output in {0, 26, 19}:
            return mlp_0_2_output == 1
        elif mlp_0_3_output in {1, 10}:
            return mlp_0_2_output == 27
        elif mlp_0_3_output in {2, 5, 15}:
            return mlp_0_2_output == 26
        elif mlp_0_3_output in {3}:
            return mlp_0_2_output == 25
        elif mlp_0_3_output in {4}:
            return mlp_0_2_output == 15
        elif mlp_0_3_output in {6, 7, 9, 20, 21, 23}:
            return mlp_0_2_output == 5
        elif mlp_0_3_output in {8}:
            return mlp_0_2_output == 17
        elif mlp_0_3_output in {11, 13}:
            return mlp_0_2_output == 3
        elif mlp_0_3_output in {12}:
            return mlp_0_2_output == 20
        elif mlp_0_3_output in {14}:
            return mlp_0_2_output == 9
        elif mlp_0_3_output in {16}:
            return mlp_0_2_output == 24
        elif mlp_0_3_output in {17, 18, 29}:
            return mlp_0_2_output == 2
        elif mlp_0_3_output in {25, 22}:
            return mlp_0_2_output == 4
        elif mlp_0_3_output in {24}:
            return mlp_0_2_output == 14
        elif mlp_0_3_output in {27}:
            return mlp_0_2_output == 12
        elif mlp_0_3_output in {28}:
            return mlp_0_2_output == 21

    attn_1_1_pattern = select_closest(mlp_0_2_outputs, mlp_0_3_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_6_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(num_mlp_0_1_output, token):
        if num_mlp_0_1_output in {0, 16, 17, 18, 28}:
            return token == "4"
        elif num_mlp_0_1_output in {1, 2}:
            return token == "0"
        elif num_mlp_0_1_output in {3, 6, 7, 8, 10, 11, 14, 25}:
            return token == ""
        elif num_mlp_0_1_output in {4}:
            return token == "2"
        elif num_mlp_0_1_output in {26, 5, 22}:
            return token == "1"
        elif num_mlp_0_1_output in {27, 9, 19, 21}:
            return token == "3"
        elif num_mlp_0_1_output in {12}:
            return token == "<pad>"
        elif num_mlp_0_1_output in {24, 29, 13, 15}:
            return token == "</s>"
        elif num_mlp_0_1_output in {20, 23}:
            return token == "<s>"

    attn_1_2_pattern = select_closest(tokens, num_mlp_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(num_mlp_0_3_output, token):
        if num_mlp_0_3_output in {0, 1, 5, 6, 11, 15, 19, 21, 23}:
            return token == ""
        elif num_mlp_0_3_output in {2, 7, 16, 17, 20, 22, 25, 28}:
            return token == "4"
        elif num_mlp_0_3_output in {24, 3}:
            return token == "1"
        elif num_mlp_0_3_output in {10, 18, 4, 14}:
            return token == "2"
        elif num_mlp_0_3_output in {8, 9, 12, 13, 27, 29}:
            return token == "3"
        elif num_mlp_0_3_output in {26}:
            return token == "0"

    attn_1_3_pattern = select_closest(tokens, num_mlp_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(mlp_0_3_output, mlp_0_2_output):
        if mlp_0_3_output in {0, 3, 6, 7, 12, 13, 28}:
            return mlp_0_2_output == 5
        elif mlp_0_3_output in {1, 11}:
            return mlp_0_2_output == 25
        elif mlp_0_3_output in {26, 2, 19}:
            return mlp_0_2_output == 1
        elif mlp_0_3_output in {17, 4, 14, 22}:
            return mlp_0_2_output == 2
        elif mlp_0_3_output in {5}:
            return mlp_0_2_output == 21
        elif mlp_0_3_output in {8}:
            return mlp_0_2_output == 17
        elif mlp_0_3_output in {9}:
            return mlp_0_2_output == 16
        elif mlp_0_3_output in {10}:
            return mlp_0_2_output == 8
        elif mlp_0_3_output in {24, 20, 15}:
            return mlp_0_2_output == 14
        elif mlp_0_3_output in {16, 21}:
            return mlp_0_2_output == 13
        elif mlp_0_3_output in {18}:
            return mlp_0_2_output == 7
        elif mlp_0_3_output in {23}:
            return mlp_0_2_output == 28
        elif mlp_0_3_output in {25}:
            return mlp_0_2_output == 4
        elif mlp_0_3_output in {27}:
            return mlp_0_2_output == 24
        elif mlp_0_3_output in {29}:
            return mlp_0_2_output == 10

    attn_1_4_pattern = select_closest(mlp_0_2_outputs, mlp_0_3_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_7_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0, 16, 15}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 25
        elif q_position in {24, 2, 4}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 16
        elif q_position in {10, 13, 5}:
            return k_position == 9
        elif q_position in {6, 7, 18, 20, 25, 29}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {9, 11, 12}:
            return k_position == 8
        elif q_position in {21, 14, 23}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 27
        elif q_position in {26, 22}:
            return k_position == 17
        elif q_position in {27}:
            return k_position == 21
        elif q_position in {28}:
            return k_position == 6

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, mlp_0_0_output):
        if position in {0, 9, 21}:
            return mlp_0_0_output == 16
        elif position in {1, 2, 8, 16, 17}:
            return mlp_0_0_output == 25
        elif position in {10, 3}:
            return mlp_0_0_output == 11
        elif position in {4, 14}:
            return mlp_0_0_output == 9
        elif position in {29, 28, 5, 23}:
            return mlp_0_0_output == 7
        elif position in {20, 6}:
            return mlp_0_0_output == 12
        elif position in {7, 15, 18, 19, 26}:
            return mlp_0_0_output == 29
        elif position in {11}:
            return mlp_0_0_output == 2
        elif position in {25, 12}:
            return mlp_0_0_output == 3
        elif position in {13}:
            return mlp_0_0_output == 23
        elif position in {24, 22}:
            return mlp_0_0_output == 6
        elif position in {27}:
            return mlp_0_0_output == 24

    attn_1_6_pattern = select_closest(mlp_0_0_outputs, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_7_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(mlp_0_3_output, token):
        if mlp_0_3_output in {0, 9, 12, 17, 18, 19, 26, 28, 29}:
            return token == "4"
        elif mlp_0_3_output in {1, 2, 25, 23}:
            return token == "0"
        elif mlp_0_3_output in {3, 5, 7, 11, 15, 20, 24}:
            return token == ""
        elif mlp_0_3_output in {4, 13}:
            return token == "2"
        elif mlp_0_3_output in {8, 21, 6, 14}:
            return token == "3"
        elif mlp_0_3_output in {10}:
            return token == "<pad>"
        elif mlp_0_3_output in {16}:
            return token == "</s>"
        elif mlp_0_3_output in {27, 22}:
            return token == "1"

    attn_1_7_pattern = select_closest(tokens, mlp_0_3_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 9, 11, 12, 13, 15, 16, 17, 18, 21, 28, 29}:
            return token == ""
        elif position in {1, 19, 7}:
            return token == "</s>"
        elif position in {2, 10, 14}:
            return token == "<s>"
        elif position in {3, 4, 5, 6, 20, 22, 23, 24, 25, 26, 27}:
            return token == "1"
        elif position in {8}:
            return token == "<pad>"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_3_output, token):
        if mlp_0_3_output in {0, 1, 4, 5, 8, 10, 11, 12, 19, 21, 23, 25, 26, 29}:
            return token == "1"
        elif mlp_0_3_output in {2, 3, 7, 9, 20, 22, 24, 27}:
            return token == "<s>"
        elif mlp_0_3_output in {6}:
            return token == "0"
        elif mlp_0_3_output in {13}:
            return token == "</s>"
        elif mlp_0_3_output in {14, 15, 16, 17, 18, 28}:
            return token == ""

    num_attn_1_1_pattern = select(tokens, mlp_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_7_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, attn_0_7_output):
        if position in {0, 16, 18, 17}:
            return attn_0_7_output == ""
        elif position in {1, 3, 14, 15, 28}:
            return attn_0_7_output == "</s>"
        elif position in {2, 4, 5, 6, 7, 9, 12, 13, 20, 22, 23, 25}:
            return attn_0_7_output == "2"
        elif position in {8, 27, 11, 21}:
            return attn_0_7_output == "1"
        elif position in {10, 19}:
            return attn_0_7_output == "0"
        elif position in {24, 26}:
            return attn_0_7_output == "3"
        elif position in {29}:
            return attn_0_7_output == "<s>"

    num_attn_1_2_pattern = select(attn_0_7_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_0_output):
        if position in {0, 1, 8, 10, 29}:
            return attn_0_0_output == "1"
        elif position in {19, 2, 11, 12}:
            return attn_0_0_output == "</s>"
        elif position in {3, 5, 6, 9, 20, 22, 24}:
            return attn_0_0_output == "2"
        elif position in {25, 4, 21}:
            return attn_0_0_output == "3"
        elif position in {7}:
            return attn_0_0_output == "0"
        elif position in {13, 14, 15, 16, 17, 18, 28}:
            return attn_0_0_output == ""
        elif position in {23}:
            return attn_0_0_output == "4"
        elif position in {26, 27}:
            return attn_0_0_output == "<s>"

    num_attn_1_3_pattern = select(attn_0_0_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, attn_0_1_output):
        if position in {0, 29, 21}:
            return attn_0_1_output == "1"
        elif position in {1, 26}:
            return attn_0_1_output == "3"
        elif position in {2, 12, 13, 14, 15, 16, 17, 18, 28}:
            return attn_0_1_output == ""
        elif position in {3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 22, 23, 24, 25, 27}:
            return attn_0_1_output == "2"

    num_attn_1_4_pattern = select(attn_0_1_outputs, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_1_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, token):
        if position in {0, 14, 15, 16, 17, 18, 28}:
            return token == ""
        elif position in {1}:
            return token == "3"
        elif position in {26, 2, 3}:
            return token == "2"
        elif position in {4, 8, 9, 20, 24, 27}:
            return token == "1"
        elif position in {5, 6, 7, 10, 12, 19, 21, 22, 23, 25, 29}:
            return token == "0"
        elif position in {11, 13}:
            return token == "<s>"

    num_attn_1_5_pattern = select(tokens, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_4_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 3, 4, 8, 9, 13, 15, 16, 17, 21, 24, 25}:
            return token == "0"
        elif mlp_0_0_output in {1, 11}:
            return token == "<s>"
        elif mlp_0_0_output in {2}:
            return token == "3"
        elif mlp_0_0_output in {5, 6, 7, 10, 12, 14, 19, 23, 26, 27}:
            return token == "1"
        elif mlp_0_0_output in {18, 28, 29}:
            return token == ""
        elif mlp_0_0_output in {20, 22}:
            return token == "2"

    num_attn_1_6_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_2_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(mlp_0_2_output, attn_0_7_output):
        if mlp_0_2_output in {
            0,
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
            22,
            25,
            28,
        }:
            return attn_0_7_output == ""
        elif mlp_0_2_output in {1, 2, 3, 4, 5, 6, 9, 20, 21, 23, 24, 26, 27, 29}:
            return attn_0_7_output == "0"

    num_attn_1_7_pattern = select(attn_0_7_outputs, mlp_0_2_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(num_mlp_0_2_output, attn_1_5_output):
        key = (num_mlp_0_2_output, attn_1_5_output)
        if key in {
            (0, "1"),
            (0, "4"),
            (0, "</s>"),
            (1, "1"),
            (1, "4"),
            (1, "</s>"),
            (2, "1"),
            (2, "4"),
            (2, "</s>"),
            (3, "1"),
            (3, "4"),
            (4, "1"),
            (4, "4"),
            (4, "</s>"),
            (5, "1"),
            (5, "4"),
            (6, "1"),
            (6, "4"),
            (6, "</s>"),
            (8, "0"),
            (8, "1"),
            (8, "2"),
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
            (10, "1"),
            (10, "4"),
            (10, "</s>"),
            (11, "1"),
            (11, "4"),
            (11, "</s>"),
            (12, "1"),
            (12, "4"),
            (12, "</s>"),
            (13, "1"),
            (13, "4"),
            (13, "</s>"),
            (14, "1"),
            (14, "4"),
            (15, "1"),
            (15, "4"),
            (15, "</s>"),
            (16, "1"),
            (16, "4"),
            (16, "</s>"),
            (17, "1"),
            (17, "4"),
            (17, "</s>"),
            (18, "1"),
            (18, "4"),
            (18, "</s>"),
            (19, "1"),
            (19, "4"),
            (19, "</s>"),
            (20, "1"),
            (20, "4"),
            (20, "</s>"),
            (21, "1"),
            (21, "4"),
            (22, "1"),
            (22, "4"),
            (22, "</s>"),
            (23, "4"),
            (24, "1"),
            (24, "4"),
            (24, "</s>"),
            (25, "0"),
            (25, "1"),
            (25, "2"),
            (25, "4"),
            (25, "</s>"),
            (25, "<s>"),
            (26, "0"),
            (26, "1"),
            (26, "2"),
            (26, "3"),
            (26, "4"),
            (26, "</s>"),
            (26, "<s>"),
            (27, "0"),
            (27, "1"),
            (27, "4"),
            (27, "</s>"),
            (28, "1"),
            (28, "4"),
            (29, "1"),
            (29, "4"),
            (29, "</s>"),
        }:
            return 11
        return 20

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(num_mlp_0_2_outputs, attn_1_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_4_output, position):
        key = (attn_1_4_output, position)
        if key in {
            ("0", 1),
            ("0", 15),
            ("0", 16),
            ("0", 18),
            ("1", 0),
            ("1", 1),
            ("1", 2),
            ("1", 8),
            ("1", 13),
            ("1", 15),
            ("1", 16),
            ("1", 17),
            ("1", 18),
            ("1", 19),
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
            ("2", 1),
            ("2", 15),
            ("2", 16),
            ("2", 18),
            ("3", 1),
            ("3", 15),
            ("3", 16),
            ("3", 18),
            ("4", 1),
            ("4", 15),
            ("4", 16),
            ("4", 17),
            ("4", 18),
            ("</s>", 1),
            ("</s>", 15),
            ("</s>", 16),
            ("</s>", 18),
            ("<s>", 1),
            ("<s>", 15),
            ("<s>", 16),
            ("<s>", 17),
            ("<s>", 18),
        }:
            return 10
        elif key in {
            ("0", 4),
            ("0", 5),
            ("0", 9),
            ("0", 17),
            ("1", 5),
            ("2", 0),
            ("2", 4),
            ("2", 5),
            ("2", 9),
            ("2", 13),
            ("2", 14),
            ("2", 17),
            ("2", 19),
            ("2", 25),
            ("2", 27),
            ("2", 29),
            ("3", 4),
            ("3", 5),
            ("3", 17),
            ("4", 4),
            ("4", 5),
            ("</s>", 0),
            ("</s>", 4),
            ("</s>", 5),
            ("</s>", 9),
            ("</s>", 12),
            ("</s>", 13),
            ("</s>", 14),
            ("</s>", 17),
            ("</s>", 19),
            ("</s>", 20),
            ("</s>", 21),
            ("</s>", 22),
            ("</s>", 23),
            ("</s>", 24),
            ("</s>", 25),
            ("</s>", 26),
            ("</s>", 27),
            ("</s>", 28),
            ("</s>", 29),
            ("<s>", 4),
            ("<s>", 5),
        }:
            return 22
        return 12

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_4_outputs, positions)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(mlp_0_3_output, mlp_0_1_output):
        key = (mlp_0_3_output, mlp_0_1_output)
        if key in {
            (0, 27),
            (1, 27),
            (2, 27),
            (6, 27),
            (7, 27),
            (8, 27),
            (11, 27),
            (12, 27),
            (13, 27),
            (14, 27),
            (15, 27),
            (17, 27),
            (18, 27),
            (19, 27),
            (20, 27),
            (21, 27),
            (22, 27),
            (24, 27),
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
            (26, 14),
            (26, 15),
            (26, 17),
            (26, 18),
            (26, 19),
            (26, 20),
            (26, 21),
            (26, 22),
            (26, 23),
            (26, 24),
            (26, 25),
            (26, 26),
            (26, 27),
            (26, 28),
            (26, 29),
            (27, 27),
            (29, 27),
        }:
            return 0
        elif key in {
            (0, 7),
            (1, 7),
            (2, 7),
            (3, 7),
            (3, 27),
            (4, 7),
            (4, 27),
            (5, 7),
            (5, 27),
            (6, 7),
            (7, 7),
            (8, 7),
            (9, 7),
            (9, 27),
            (10, 7),
            (11, 7),
            (12, 7),
            (13, 7),
            (14, 7),
            (15, 7),
            (16, 7),
            (17, 7),
            (18, 7),
            (19, 7),
            (20, 7),
            (21, 7),
            (22, 7),
            (23, 7),
            (23, 27),
            (24, 7),
            (25, 7),
            (27, 7),
            (29, 7),
        }:
            return 15
        return 11

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(mlp_0_3_outputs, mlp_0_1_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_0_1_output):
        key = attn_0_1_output
        if key in {""}:
            return 23
        elif key in {"</s>"}:
            return 24
        return 19

    mlp_1_3_outputs = [mlp_1_3(k0) for k0 in attn_0_1_outputs]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_6_output):
        key = num_attn_0_6_output
        if key in {0}:
            return 18
        return 19

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_6_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_1_7_output):
        key = (num_attn_1_0_output, num_attn_1_7_output)
        return 0

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_0_0_output):
        key = num_attn_0_0_output
        if key in {0}:
            return 28
        return 6

    num_mlp_1_2_outputs = [num_mlp_1_2(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_6_output, num_attn_1_7_output):
        key = (num_attn_1_6_output, num_attn_1_7_output)
        return 12

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "<s>"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"<s>", "</s>", "4", "3"}:
            return k_token == ""

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_4_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_num_mlp_1_2_output, k_num_mlp_1_2_output):
        if q_num_mlp_1_2_output in {0, 3}:
            return k_num_mlp_1_2_output == 21
        elif q_num_mlp_1_2_output in {1, 15}:
            return k_num_mlp_1_2_output == 8
        elif q_num_mlp_1_2_output in {2}:
            return k_num_mlp_1_2_output == 10
        elif q_num_mlp_1_2_output in {18, 4}:
            return k_num_mlp_1_2_output == 3
        elif q_num_mlp_1_2_output in {21, 5}:
            return k_num_mlp_1_2_output == 14
        elif q_num_mlp_1_2_output in {6}:
            return k_num_mlp_1_2_output == 9
        elif q_num_mlp_1_2_output in {22, 7}:
            return k_num_mlp_1_2_output == 17
        elif q_num_mlp_1_2_output in {8}:
            return k_num_mlp_1_2_output == 2
        elif q_num_mlp_1_2_output in {9, 26, 20, 29}:
            return k_num_mlp_1_2_output == 23
        elif q_num_mlp_1_2_output in {10, 11}:
            return k_num_mlp_1_2_output == 27
        elif q_num_mlp_1_2_output in {17, 19, 12}:
            return k_num_mlp_1_2_output == 28
        elif q_num_mlp_1_2_output in {13}:
            return k_num_mlp_1_2_output == 0
        elif q_num_mlp_1_2_output in {24, 14}:
            return k_num_mlp_1_2_output == 18
        elif q_num_mlp_1_2_output in {16, 28, 23}:
            return k_num_mlp_1_2_output == 6
        elif q_num_mlp_1_2_output in {25}:
            return k_num_mlp_1_2_output == 7
        elif q_num_mlp_1_2_output in {27}:
            return k_num_mlp_1_2_output == 26

    attn_2_1_pattern = select_closest(
        num_mlp_1_2_outputs, num_mlp_1_2_outputs, predicate_2_1
    )
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_7_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "</s>"
        elif q_token in {"3", "1"}:
            return k_token == ""
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"</s>"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_4_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, mlp_0_2_output):
        if token in {"0"}:
            return mlp_0_2_output == 17
        elif token in {"1"}:
            return mlp_0_2_output == 4
        elif token in {"2"}:
            return mlp_0_2_output == 12
        elif token in {"3"}:
            return mlp_0_2_output == 2
        elif token in {"4"}:
            return mlp_0_2_output == 15
        elif token in {"</s>"}:
            return mlp_0_2_output == 27
        elif token in {"<s>"}:
            return mlp_0_2_output == 26

    attn_2_3_pattern = select_closest(mlp_0_2_outputs, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_6_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(mlp_1_3_output, token):
        if mlp_1_3_output in {0, 8, 10, 13, 26}:
            return token == "3"
        elif mlp_1_3_output in {1, 2, 4, 5, 6, 7, 19, 24, 27}:
            return token == "</s>"
        elif mlp_1_3_output in {3, 11, 12, 14, 15, 16, 17, 18, 20, 22}:
            return token == ""
        elif mlp_1_3_output in {9}:
            return token == "2"
        elif mlp_1_3_output in {28, 21}:
            return token == "<s>"
        elif mlp_1_3_output in {29, 23}:
            return token == "0"
        elif mlp_1_3_output in {25}:
            return token == "4"

    attn_2_4_pattern = select_closest(tokens, mlp_1_3_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_5_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(mlp_0_2_output, mlp_0_3_output):
        if mlp_0_2_output in {0, 25, 11, 20}:
            return mlp_0_3_output == 28
        elif mlp_0_2_output in {1, 27, 13}:
            return mlp_0_3_output == 4
        elif mlp_0_2_output in {2, 6}:
            return mlp_0_3_output == 6
        elif mlp_0_2_output in {10, 3}:
            return mlp_0_3_output == 29
        elif mlp_0_2_output in {24, 4, 12, 7}:
            return mlp_0_3_output == 23
        elif mlp_0_2_output in {5, 14}:
            return mlp_0_3_output == 26
        elif mlp_0_2_output in {8, 16}:
            return mlp_0_3_output == 25
        elif mlp_0_2_output in {9, 28, 29}:
            return mlp_0_3_output == 9
        elif mlp_0_2_output in {15}:
            return mlp_0_3_output == 12
        elif mlp_0_2_output in {17, 22}:
            return mlp_0_3_output == 22
        elif mlp_0_2_output in {18}:
            return mlp_0_3_output == 3
        elif mlp_0_2_output in {19}:
            return mlp_0_3_output == 18
        elif mlp_0_2_output in {21}:
            return mlp_0_3_output == 5
        elif mlp_0_2_output in {23}:
            return mlp_0_3_output == 24
        elif mlp_0_2_output in {26}:
            return mlp_0_3_output == 7

    attn_2_5_pattern = select_closest(mlp_0_3_outputs, mlp_0_2_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_0_3_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
        if q_token in {"<s>", "0"}:
            return k_token == "</s>"
        elif q_token in {"1"}:
            return k_token == ""
        elif q_token in {"2"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"4"}:
            return k_token == "3"
        elif q_token in {"</s>"}:
            return k_token == "<pad>"

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_1_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_token, k_token):
        if q_token in {"4", "0"}:
            return k_token == "3"
        elif q_token in {"<s>", "1"}:
            return k_token == ""
        elif q_token in {"2", "3"}:
            return k_token == "4"
        elif q_token in {"</s>"}:
            return k_token == "<s>"

    attn_2_7_pattern = select_closest(tokens, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_4_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_3_output, attn_1_7_output):
        if mlp_0_3_output in {0, 1, 3, 8, 11, 19, 21, 25}:
            return attn_1_7_output == "1"
        elif mlp_0_3_output in {2, 4, 5, 6, 7, 9, 20, 22, 24, 27}:
            return attn_1_7_output == "0"
        elif mlp_0_3_output in {10}:
            return attn_1_7_output == "</s>"
        elif mlp_0_3_output in {12, 13, 14, 15, 16, 17, 18, 28, 29}:
            return attn_1_7_output == ""
        elif mlp_0_3_output in {26, 23}:
            return attn_1_7_output == "2"

    num_attn_2_0_pattern = select(attn_1_7_outputs, mlp_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_2_output, attn_1_0_output):
        if attn_1_2_output in {"0"}:
            return attn_1_0_output == "</s>"
        elif attn_1_2_output in {"1"}:
            return attn_1_0_output == "0"
        elif attn_1_2_output in {"<s>", "4", "3", "2", "</s>"}:
            return attn_1_0_output == "1"

    num_attn_2_1_pattern = select(attn_1_0_outputs, attn_1_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_2_output, attn_0_2_output):
        if mlp_0_2_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
            11,
            13,
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
        }:
            return attn_0_2_output == "0"
        elif mlp_0_2_output in {7, 12, 14, 15, 18, 28, 29}:
            return attn_0_2_output == ""

    num_attn_2_2_pattern = select(attn_0_2_outputs, mlp_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_7_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_2_output, attn_0_0_output):
        if attn_1_2_output in {"0", "1"}:
            return attn_0_0_output == "1"
        elif attn_1_2_output in {"2", "3"}:
            return attn_0_0_output == "2"
        elif attn_1_2_output in {"</s>", "4", "<s>"}:
            return attn_0_0_output == "0"

    num_attn_2_3_pattern = select(attn_0_0_outputs, attn_1_2_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_2_output, attn_1_0_output):
        if attn_1_2_output in {"<s>", "4", "1", "</s>", "0"}:
            return attn_1_0_output == "2"
        elif attn_1_2_output in {"2"}:
            return attn_1_0_output == "1"
        elif attn_1_2_output in {"3"}:
            return attn_1_0_output == "<s>"

    num_attn_2_4_pattern = select(attn_1_0_outputs, attn_1_2_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(mlp_0_0_output, attn_0_5_output):
        if mlp_0_0_output in {0, 4, 5, 7, 12, 17, 18, 20, 21, 22, 24, 27, 28, 29}:
            return attn_0_5_output == ""
        elif mlp_0_0_output in {1, 2, 6, 8, 9, 10, 13, 14, 15, 16, 19, 23, 25, 26}:
            return attn_0_5_output == "2"
        elif mlp_0_0_output in {11, 3}:
            return attn_0_5_output == "0"

    num_attn_2_5_pattern = select(attn_0_5_outputs, mlp_0_0_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_5_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_2_output, attn_1_0_output):
        if attn_1_2_output in {"<s>", "1", "2", "</s>", "0"}:
            return attn_1_0_output == "3"
        elif attn_1_2_output in {"3"}:
            return attn_1_0_output == "0"
        elif attn_1_2_output in {"4"}:
            return attn_1_0_output == ""

    num_attn_2_6_pattern = select(attn_1_0_outputs, attn_1_2_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_1_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(mlp_0_2_output, attn_1_6_output):
        if mlp_0_2_output in {0, 2, 4, 5, 6, 8, 16, 25}:
            return attn_1_6_output == "1"
        elif mlp_0_2_output in {27, 1, 19}:
            return attn_1_6_output == "<s>"
        elif mlp_0_2_output in {
            3,
            7,
            9,
            10,
            12,
            13,
            14,
            17,
            18,
            21,
            22,
            23,
            24,
            26,
            28,
            29,
        }:
            return attn_1_6_output == ""
        elif mlp_0_2_output in {11, 20}:
            return attn_1_6_output == "</s>"
        elif mlp_0_2_output in {15}:
            return attn_1_6_output == "<pad>"

    num_attn_2_7_pattern = select(attn_1_6_outputs, mlp_0_2_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_2_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_0_output, position):
        key = (attn_2_0_output, position)
        if key in {
            ("0", 2),
            ("0", 5),
            ("0", 6),
            ("0", 8),
            ("0", 9),
            ("0", 16),
            ("0", 17),
            ("1", 2),
            ("1", 5),
            ("1", 6),
            ("1", 8),
            ("1", 9),
            ("1", 16),
            ("1", 17),
            ("2", 2),
            ("2", 5),
            ("2", 6),
            ("2", 8),
            ("2", 9),
            ("2", 16),
            ("2", 17),
            ("3", 2),
            ("3", 5),
            ("3", 6),
            ("3", 8),
            ("3", 9),
            ("3", 16),
            ("3", 17),
            ("4", 0),
            ("4", 2),
            ("4", 5),
            ("4", 6),
            ("4", 8),
            ("4", 9),
            ("4", 16),
            ("4", 17),
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
            ("</s>", 2),
            ("</s>", 5),
            ("</s>", 6),
            ("</s>", 8),
            ("</s>", 9),
            ("</s>", 16),
            ("</s>", 17),
            ("<s>", 2),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 16),
            ("<s>", 17),
        }:
            return 15
        elif key in {
            ("0", 10),
            ("0", 13),
            ("1", 10),
            ("1", 13),
            ("2", 10),
            ("2", 13),
            ("3", 10),
            ("3", 13),
            ("4", 10),
            ("4", 13),
            ("</s>", 10),
            ("</s>", 13),
            ("<s>", 10),
            ("<s>", 13),
        }:
            return 7
        elif key in {
            ("0", 7),
            ("0", 18),
            ("1", 7),
            ("2", 7),
            ("2", 18),
            ("3", 7),
            ("3", 18),
            ("4", 7),
            ("4", 18),
            ("</s>", 7),
            ("</s>", 18),
            ("<s>", 7),
            ("<s>", 18),
        }:
            return 14
        return 19

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position, attn_2_3_output):
        key = (position, attn_2_3_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (14, "1"),
            (16, "1"),
            (19, "1"),
            (21, "1"),
            (23, "1"),
            (24, "1"),
            (26, "0"),
            (26, "1"),
            (26, "2"),
            (29, "1"),
        }:
            return 12
        elif key in {(1, "<s>"), (2, "<s>"), (26, "</s>"), (26, "<s>")}:
            return 10
        return 22

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(positions, attn_2_3_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(mlp_0_0_output, attn_2_6_output):
        key = (mlp_0_0_output, attn_2_6_output)
        if key in {
            (0, "</s>"),
            (0, "<s>"),
            (2, "2"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (4, "</s>"),
            (4, "<s>"),
            (5, "<s>"),
            (8, "</s>"),
            (12, "</s>"),
            (12, "<s>"),
            (13, "<s>"),
            (15, "<s>"),
            (16, "0"),
            (16, "1"),
            (16, "2"),
            (16, "3"),
            (16, "4"),
            (16, "</s>"),
            (16, "<s>"),
            (17, "<s>"),
            (18, "<s>"),
            (19, "</s>"),
            (19, "<s>"),
            (20, "</s>"),
            (20, "<s>"),
            (21, "</s>"),
            (21, "<s>"),
            (22, "</s>"),
            (22, "<s>"),
            (23, "1"),
            (23, "</s>"),
            (23, "<s>"),
            (24, "</s>"),
            (24, "<s>"),
            (27, "</s>"),
            (27, "<s>"),
            (28, "</s>"),
            (28, "<s>"),
        }:
            return 15
        elif key in {
            (0, "1"),
            (1, "1"),
            (1, "2"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "1"),
            (3, "1"),
            (3, "2"),
            (3, "4"),
            (3, "<s>"),
            (4, "1"),
            (7, "1"),
            (8, "1"),
            (11, "0"),
            (11, "1"),
            (11, "2"),
            (11, "3"),
            (11, "4"),
            (11, "</s>"),
            (11, "<s>"),
            (12, "1"),
            (15, "1"),
            (19, "1"),
            (20, "1"),
            (21, "1"),
            (22, "1"),
            (24, "1"),
            (26, "1"),
            (27, "1"),
            (28, "1"),
        }:
            return 0
        elif key in {(5, "</s>"), (13, "</s>"), (14, "</s>"), (26, "</s>")}:
            return 23
        return 9

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, attn_2_6_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(position):
        key = position
        if key in {1, 2, 3, 4, 15, 26}:
            return 24
        elif key in {12}:
            return 25
        return 13

    mlp_2_3_outputs = [mlp_2_3(k0) for k0 in positions]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_7_output, num_attn_2_7_output):
        key = (num_attn_1_7_output, num_attn_2_7_output)
        return 20

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_2_7_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_7_output, num_attn_1_1_output):
        key = (num_attn_1_7_output, num_attn_1_1_output)
        return 0

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_7_output):
        key = num_attn_1_7_output
        return 16

    num_mlp_2_2_outputs = [num_mlp_2_2(k0) for k0 in num_attn_1_7_outputs]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_0_6_output):
        key = num_attn_0_6_output
        if key in {0}:
            return 8
        return 26

    num_mlp_2_3_outputs = [num_mlp_2_3(k0) for k0 in num_attn_0_6_outputs]
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


print(run(["<s>", "3", "4", "0", "1", "3", "0", "</s>"]))
