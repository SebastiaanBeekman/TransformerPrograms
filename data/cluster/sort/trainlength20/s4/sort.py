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
        "output/length/rasp/sort/trainlength20/s4/sort_weights.csv",
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
        if q_position in {0, 1, 3, 8}:
            return k_position == 5
        elif q_position in {2, 6, 7, 15, 16, 17, 18, 19}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {13, 5}:
            return k_position == 3
        elif q_position in {9, 10}:
            return k_position == 12
        elif q_position in {11, 12}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {20, 29}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 24
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 14
        elif q_position in {24}:
            return k_position == 29
        elif q_position in {25, 26}:
            return k_position == 20
        elif q_position in {27}:
            return k_position == 27
        elif q_position in {28}:
            return k_position == 17

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 2, 19}:
            return token == "2"
        elif position in {1, 18}:
            return token == "0"
        elif position in {3, 4}:
            return token == "1"
        elif position in {5, 7, 9, 10, 11, 13, 14, 17}:
            return token == "3"
        elif position in {6, 8, 12, 15, 16}:
            return token == "4"
        elif position in {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 1, 11}:
            return token == "1"
        elif position in {2, 12, 5}:
            return token == "0"
        elif position in {3, 7, 10, 15, 16, 19}:
            return token == "2"
        elif position in {4, 6, 9, 14, 17, 18}:
            return token == "3"
        elif position in {8, 13}:
            return token == "4"
        elif position in {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 1
        elif q_position in {1, 7}:
            return k_position == 2
        elif q_position in {16, 2, 18}:
            return k_position == 3
        elif q_position in {19, 3}:
            return k_position == 4
        elif q_position in {4, 28}:
            return k_position == 5
        elif q_position in {12, 5}:
            return k_position == 6
        elif q_position in {6, 22}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 8
        elif q_position in {17, 23}:
            return k_position == 16
        elif q_position in {20}:
            return k_position == 23
        elif q_position in {24, 21}:
            return k_position == 22
        elif q_position in {25, 27, 29}:
            return k_position == 18
        elif q_position in {26}:
            return k_position == 12

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17}:
            return token == "3"
        elif position in {1, 2, 18, 15}:
            return token == "1"
        elif position in {4}:
            return token == "4"
        elif position in {16, 19, 21}:
            return token == "2"
        elif position in {20, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 11, 3, 6}:
            return token == "1"
        elif position in {1, 5, 7, 10, 13, 14, 19}:
            return token == "3"
        elif position in {2}:
            return token == "0"
        elif position in {8, 16, 4, 15}:
            return token == "4"
        elif position in {9, 12, 17}:
            return token == "2"
        elif position in {18}:
            return token == "<s>"
        elif position in {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0, 6, 7, 8, 28}:
            return token == "4"
        elif position in {1, 11, 12, 14, 20, 21, 22, 23, 24, 25, 26, 27, 29}:
            return token == ""
        elif position in {2, 5, 16, 17, 19}:
            return token == "3"
        elif position in {9, 10, 3}:
            return token == "1"
        elif position in {18, 4, 13}:
            return token == "2"
        elif position in {15}:
            return token == "<pad>"

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0}:
            return token == "1"
        elif position in {
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
            17,
            18,
            19,
        }:
            return token == "3"
        elif position in {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {19, 5, 6}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {9, 28}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {11, 12}:
            return k_position == 15
        elif q_position in {13}:
            return k_position == 17
        elif q_position in {14}:
            return k_position == 18
        elif q_position in {15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 27
        elif q_position in {17}:
            return k_position == 29
        elif q_position in {18}:
            return k_position == 26
        elif q_position in {20, 21, 22, 24, 25, 26, 27}:
            return k_position == 3
        elif q_position in {23}:
            return k_position == 11
        elif q_position in {29}:
            return k_position == 16

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 11, 13, 14, 16, 18, 19}:
            return token == "1"
        elif position in {1, 2, 3, 5, 20, 22, 24, 27, 28}:
            return token == ""
        elif position in {26, 29, 4, 21}:
            return token == "<pad>"
        elif position in {8, 6}:
            return token == "<s>"
        elif position in {7}:
            return token == "</s>"
        elif position in {9, 12, 17, 15}:
            return token == "0"
        elif position in {10}:
            return token == "2"
        elif position in {23}:
            return token == "3"
        elif position in {25}:
            return token == "4"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 27, 20, 22}:
            return token == "1"
        elif position in {1}:
            return token == "</s>"
        elif position in {2, 3, 4, 5, 6, 19, 21, 23, 24, 25, 26, 28, 29}:
            return token == "0"
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}:
            return token == ""

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1, 2, 3, 4, 18, 19, 24, 25, 29}:
            return token == "1"
        elif position in {5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 26, 27, 28}:
            return token == ""
        elif position in {12}:
            return token == "</s>"
        elif position in {13}:
            return token == "<s>"
        elif position in {16, 17, 14, 15}:
            return token == "4"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 2, 3, 5, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {4}:
            return token == "<pad>"
        elif position in {8, 6, 7}:
            return token == "</s>"
        elif position in {9, 10, 11, 12, 13, 14}:
            return token == "4"
        elif position in {16, 18, 15}:
            return token == "1"
        elif position in {17, 19}:
            return token == "2"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0}:
            return token == "2"
        elif position in {19, 1, 2, 18}:
            return token == "0"
        elif position in {3, 4, 5, 6, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {8, 10, 12, 7}:
            return token == "</s>"
        elif position in {9, 11}:
            return token == "<s>"
        elif position in {16, 13, 14, 15}:
            return token == "4"
        elif position in {17}:
            return token == "1"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0, 9, 22}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {3, 4, 6, 7}:
            return k_position == 9
        elif q_position in {8, 29, 5}:
            return k_position == 10
        elif q_position in {26, 25, 10, 11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {23, 14, 15}:
            return k_position == 17
        elif q_position in {16}:
            return k_position == 19
        elif q_position in {17, 18}:
            return k_position == 28
        elif q_position in {19}:
            return k_position == 7
        elif q_position in {20}:
            return k_position == 15
        elif q_position in {21}:
            return k_position == 4
        elif q_position in {24}:
            return k_position == 21
        elif q_position in {27, 28}:
            return k_position == 1

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1}:
            return token == "<s>"
        elif position in {2, 3, 4, 5}:
            return token == "1"
        elif position in {19, 6, 7}:
            return token == "0"
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

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {1, 2, 19}:
            return 25
        elif key in {0}:
            return 14
        elif key in {3}:
            return 19
        return 1

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_7_output):
        key = attn_0_7_output
        return 2

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in attn_0_7_outputs]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_6_output, position):
        key = (attn_0_6_output, position)
        if key in {
            ("0", 0),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("0", 6),
            ("1", 0),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 20),
            ("1", 21),
            ("1", 24),
            ("1", 26),
            ("1", 27),
            ("1", 28),
            ("1", 29),
            ("2", 0),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("3", 0),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("4", 0),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 6),
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
            ("</s>", 0),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 5),
            ("</s>", 6),
            ("<s>", 0),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 6),
        }:
            return 23
        return 24

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_6_outputs, positions)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_1_output):
        key = attn_0_1_output
        return 14

    mlp_0_3_outputs = [mlp_0_3(k0) for k0 in attn_0_1_outputs]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 13

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 25

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_6_output, num_attn_0_5_output):
        key = (num_attn_0_6_output, num_attn_0_5_output)
        return 12

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_6_output):
        key = num_attn_0_6_output
        return 13

    num_mlp_0_3_outputs = [num_mlp_0_3(k0) for k0 in num_attn_0_6_outputs]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 2, 5, 6, 7, 11, 16}:
            return token == "4"
        elif position in {1, 12, 25, 15}:
            return token == "0"
        elif position in {3, 4, 13, 14, 19}:
            return token == "2"
        elif position in {8, 9, 17}:
            return token == "1"
        elif position in {10, 18}:
            return token == "3"
        elif position in {20, 21, 23, 24, 26, 27, 28, 29}:
            return token == ""
        elif position in {22}:
            return token == "</s>"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_attn_0_4_output, k_attn_0_4_output):
        if q_attn_0_4_output in {"0"}:
            return k_attn_0_4_output == "3"
        elif q_attn_0_4_output in {"2", "1"}:
            return k_attn_0_4_output == ""
        elif q_attn_0_4_output in {"3"}:
            return k_attn_0_4_output == "1"
        elif q_attn_0_4_output in {"4", "<s>"}:
            return k_attn_0_4_output == "</s>"
        elif q_attn_0_4_output in {"</s>"}:
            return k_attn_0_4_output == "4"

    attn_1_1_pattern = select_closest(attn_0_4_outputs, attn_0_4_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_7_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, token):
        if position in {0, 2, 3, 4, 5, 6, 8, 10, 16, 17}:
            return token == "4"
        elif position in {1, 13, 9}:
            return token == "1"
        elif position in {19, 15, 29, 7}:
            return token == "3"
        elif position in {26, 18, 11, 14}:
            return token == "2"
        elif position in {25, 27, 12, 23}:
            return token == "0"
        elif position in {24, 20, 21, 22}:
            return token == ""
        elif position in {28}:
            return token == "</s>"

    attn_1_2_pattern = select_closest(tokens, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_2_output, token):
        if mlp_0_2_output in {0, 2, 12, 13, 19, 20}:
            return token == "1"
        elif mlp_0_2_output in {1, 14, 15, 16, 21, 23, 24, 26}:
            return token == "4"
        elif mlp_0_2_output in {8, 11, 3}:
            return token == "3"
        elif mlp_0_2_output in {4, 5, 6, 9, 10, 18, 27, 28}:
            return token == ""
        elif mlp_0_2_output in {7}:
            return token == "<pad>"
        elif mlp_0_2_output in {17, 25}:
            return token == "0"
        elif mlp_0_2_output in {29, 22}:
            return token == "2"

    attn_1_3_pattern = select_closest(tokens, mlp_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, token):
        if position in {0, 5}:
            return token == "2"
        elif position in {1, 16, 20, 21, 22, 23, 24, 26, 27, 28, 29}:
            return token == ""
        elif position in {2, 9, 14, 15, 25}:
            return token == "0"
        elif position in {18, 3, 4}:
            return token == "1"
        elif position in {6, 7, 8, 10, 17}:
            return token == "4"
        elif position in {19, 11, 12, 13}:
            return token == "3"

    attn_1_4_pattern = select_closest(tokens, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_7_output, token):
        if attn_0_7_output in {"<s>", "1", "0", "</s>"}:
            return token == "4"
        elif attn_0_7_output in {"2"}:
            return token == "0"
        elif attn_0_7_output in {"3"}:
            return token == "2"
        elif attn_0_7_output in {"4"}:
            return token == "1"

    attn_1_5_pattern = select_closest(tokens, attn_0_7_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_3_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, token):
        if position in {0, 25, 22}:
            return token == "0"
        elif position in {1}:
            return token == "1"
        elif position in {2, 3, 14}:
            return token == "2"
        elif position in {4, 5, 7, 8, 9, 10, 12, 13, 16, 17, 18, 19}:
            return token == "3"
        elif position in {6}:
            return token == "4"
        elif position in {11, 15, 20, 21, 23, 24, 26, 27, 28, 29}:
            return token == ""

    attn_1_6_pattern = select_closest(tokens, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_position, k_position):
        if q_position in {0, 20}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 18
        elif q_position in {2, 13, 14}:
            return k_position == 19
        elif q_position in {8, 3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {29, 5}:
            return k_position == 7
        elif q_position in {6, 7, 16, 17, 18}:
            return k_position == 8
        elif q_position in {9, 19}:
            return k_position == 13
        elif q_position in {10, 11, 12}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 10
        elif q_position in {26, 21}:
            return k_position == 6
        elif q_position in {22}:
            return k_position == 27
        elif q_position in {23}:
            return k_position == 14
        elif q_position in {24, 28}:
            return k_position == 16
        elif q_position in {25, 27}:
            return k_position == 3

    attn_1_7_pattern = select_closest(positions, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_0_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_2_output, attn_0_1_output):
        if mlp_0_2_output in {0, 3, 8, 10, 12, 13, 15, 16, 17, 18, 21, 22, 24, 25, 27}:
            return attn_0_1_output == "0"
        elif mlp_0_2_output in {1, 2, 4, 5, 6, 7, 9, 11, 14, 19, 20, 23, 26, 28, 29}:
            return attn_0_1_output == "2"

    num_attn_1_0_pattern = select(attn_0_1_outputs, mlp_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_5_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, attn_0_2_output):
        if position in {0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}:
            return attn_0_2_output == ""
        elif position in {1, 2, 3, 4, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return attn_0_2_output == "0"
        elif position in {5}:
            return attn_0_2_output == "</s>"
        elif position in {19}:
            return attn_0_2_output == "1"

    num_attn_1_1_pattern = select(attn_0_2_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, attn_0_2_output):
        if position in {0, 14, 15, 16, 17, 18}:
            return attn_0_2_output == ""
        elif position in {1, 2, 12}:
            return attn_0_2_output == "<s>"
        elif position in {3, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return attn_0_2_output == "2"
        elif position in {19, 4}:
            return attn_0_2_output == "3"
        elif position in {11, 13}:
            return attn_0_2_output == "</s>"

    num_attn_1_2_pattern = select(attn_0_2_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_1_output):
        if position in {0, 10, 11, 12, 13, 14, 15, 16, 17, 18}:
            return attn_0_1_output == ""
        elif position in {1, 2, 3, 21, 28}:
            return attn_0_1_output == "2"
        elif position in {4, 5, 6, 7, 20, 22, 23, 24, 26, 27, 29}:
            return attn_0_1_output == "1"
        elif position in {8}:
            return attn_0_1_output == "0"
        elif position in {9}:
            return attn_0_1_output == "<s>"
        elif position in {25, 19}:
            return attn_0_1_output == "3"

    num_attn_1_3_pattern = select(attn_0_1_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_7_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(mlp_0_1_output, token):
        if mlp_0_1_output in {
            0,
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
            17,
            18,
            21,
            22,
            27,
        }:
            return token == "<s>"
        elif mlp_0_1_output in {1, 6, 7, 8, 20, 24, 25, 26, 29}:
            return token == "2"
        elif mlp_0_1_output in {9}:
            return token == "0"
        elif mlp_0_1_output in {16, 19, 28, 23}:
            return token == "</s>"

    num_attn_1_4_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_3_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, attn_0_5_output):
        if position in {0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 28}:
            return attn_0_5_output == ""
        elif position in {1, 19}:
            return attn_0_5_output == "1"
        elif position in {2, 3, 4, 5, 20, 21, 23, 24, 25, 26, 27, 29}:
            return attn_0_5_output == "0"

    num_attn_1_5_pattern = select(attn_0_5_outputs, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_5_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, attn_0_2_output):
        if position in {0, 10, 11, 12, 13, 14, 15, 16, 17, 18}:
            return attn_0_2_output == ""
        elif position in {1, 9}:
            return attn_0_2_output == "</s>"
        elif position in {2, 6, 20, 22, 23, 24, 26, 27, 28, 29}:
            return attn_0_2_output == "1"
        elif position in {3, 4, 5, 7}:
            return attn_0_2_output == "0"
        elif position in {8}:
            return attn_0_2_output == "<s>"
        elif position in {19}:
            return attn_0_2_output == "3"
        elif position in {25, 21}:
            return attn_0_2_output == "2"

    num_attn_1_6_pattern = select(attn_0_2_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_1_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, attn_0_2_output):
        if position in {0, 12, 13, 14, 15, 16, 17, 18}:
            return attn_0_2_output == ""
        elif position in {1, 21, 25}:
            return attn_0_2_output == "2"
        elif position in {2, 3, 4, 5, 6, 7, 8, 19, 20, 22, 23, 24, 26, 27, 28, 29}:
            return attn_0_2_output == "1"
        elif position in {9, 10}:
            return attn_0_2_output == "0"
        elif position in {11}:
            return attn_0_2_output == "</s>"

    num_attn_1_7_pattern = select(attn_0_2_outputs, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_1_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_1_output):
        key = (position, attn_1_1_output)
        if key in {
            (0, "1"),
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
            (19, "0"),
            (19, "1"),
            (19, "2"),
            (19, "3"),
            (19, "4"),
            (19, "</s>"),
            (19, "<s>"),
            (20, "0"),
            (20, "1"),
            (20, "2"),
            (21, "0"),
            (21, "1"),
            (21, "2"),
            (22, "0"),
            (22, "1"),
            (22, "2"),
            (23, "0"),
            (23, "1"),
            (23, "2"),
            (23, "3"),
            (23, "<s>"),
            (24, "0"),
            (24, "1"),
            (24, "2"),
            (25, "0"),
            (25, "1"),
            (25, "2"),
            (25, "3"),
            (25, "4"),
            (25, "</s>"),
            (25, "<s>"),
            (26, "0"),
            (26, "1"),
            (26, "2"),
            (27, "0"),
            (27, "1"),
            (27, "2"),
            (28, "0"),
            (28, "1"),
            (28, "2"),
            (29, "0"),
            (29, "1"),
        }:
            return 7
        elif key in {
            (0, "0"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "</s>"),
            (4, "<s>"),
            (20, "3"),
            (20, "</s>"),
            (20, "<s>"),
            (21, "3"),
            (21, "</s>"),
            (22, "3"),
            (22, "</s>"),
            (23, "</s>"),
            (24, "</s>"),
            (26, "3"),
            (26, "</s>"),
            (26, "<s>"),
            (27, "3"),
            (27, "</s>"),
            (27, "<s>"),
            (28, "3"),
            (28, "</s>"),
            (28, "<s>"),
            (29, "2"),
            (29, "</s>"),
        }:
            return 28
        return 10

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_1_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position):
        key = position
        if key in {1, 2, 3, 4, 19, 25}:
            return 25
        return 24

    mlp_1_1_outputs = [mlp_1_1(k0) for k0 in positions]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(mlp_0_0_output, attn_1_1_output):
        key = (mlp_0_0_output, attn_1_1_output)
        if key in {
            (0, "1"),
            (2, "1"),
            (7, "1"),
            (8, "1"),
            (9, "1"),
            (10, "1"),
            (11, "1"),
            (12, "1"),
            (13, "1"),
            (14, "1"),
            (15, "1"),
            (16, "1"),
            (17, "1"),
            (18, "1"),
            (19, "1"),
            (20, "1"),
            (21, "1"),
            (22, "1"),
            (23, "1"),
            (24, "1"),
            (25, "1"),
            (26, "1"),
            (27, "1"),
            (28, "1"),
            (29, "1"),
        }:
            return 21
        elif key in {
            (2, "4"),
            (4, "4"),
            (4, "</s>"),
            (5, "4"),
            (6, "4"),
            (16, "4"),
            (17, "4"),
            (18, "4"),
            (22, "4"),
            (23, "4"),
            (23, "</s>"),
            (24, "4"),
            (25, "4"),
            (26, "4"),
            (27, "4"),
        }:
            return 11
        return 27

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, attn_1_1_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_6_output, num_mlp_0_2_output):
        key = (attn_1_6_output, num_mlp_0_2_output)
        return 13

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_6_outputs, num_mlp_0_2_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_7_output, num_attn_1_1_output):
        key = (num_attn_1_7_output, num_attn_1_1_output)
        return 8

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_0_output):
        key = num_attn_0_0_output
        if key in {0}:
            return 21
        return 7

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_6_output, num_attn_1_5_output):
        key = (num_attn_1_6_output, num_attn_1_5_output)
        return 21

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_6_output, num_attn_1_1_output):
        key = (num_attn_1_6_output, num_attn_1_1_output)
        return 11

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 17, 26}:
            return token == "2"
        elif mlp_0_0_output in {1, 3, 5, 6, 9, 15, 19}:
            return token == "</s>"
        elif mlp_0_0_output in {2}:
            return token == "<s>"
        elif mlp_0_0_output in {4, 7, 8, 10, 11, 12, 14, 16, 20, 23, 24, 27, 28}:
            return token == ""
        elif mlp_0_0_output in {13, 18, 22, 25, 29}:
            return token == "0"
        elif mlp_0_0_output in {21}:
            return token == "4"

    attn_2_0_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_6_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 12, 22, 26, 29}:
            return token == "1"
        elif mlp_0_0_output in {1, 7, 8, 11, 15, 16, 18, 24, 28}:
            return token == ""
        elif mlp_0_0_output in {2, 3, 5, 14}:
            return token == "3"
        elif mlp_0_0_output in {17, 4, 13, 25}:
            return token == "0"
        elif mlp_0_0_output in {6}:
            return token == "<pad>"
        elif mlp_0_0_output in {9, 27, 23}:
            return token == "<s>"
        elif mlp_0_0_output in {10}:
            return token == "4"
        elif mlp_0_0_output in {19, 20}:
            return token == "2"
        elif mlp_0_0_output in {21}:
            return token == "</s>"

    attn_2_1_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, attn_1_6_output):
        if token in {"0"}:
            return attn_1_6_output == "4"
        elif token in {"1"}:
            return attn_1_6_output == ""
        elif token in {"2", "</s>"}:
            return attn_1_6_output == "<s>"
        elif token in {"3"}:
            return attn_1_6_output == "2"
        elif token in {"4"}:
            return attn_1_6_output == "3"
        elif token in {"<s>"}:
            return attn_1_6_output == "</s>"

    attn_2_2_pattern = select_closest(attn_1_6_outputs, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "</s>"
        elif attn_0_4_output in {"1"}:
            return token == "2"
        elif attn_0_4_output in {"2", "3"}:
            return token == "1"
        elif attn_0_4_output in {"4"}:
            return token == "<s>"
        elif attn_0_4_output in {"<s>", "</s>"}:
            return token == "<pad>"

    attn_2_3_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 25, 27}:
            return position == 1
        elif mlp_0_0_output in {1}:
            return position == 18
        elif mlp_0_0_output in {2}:
            return position == 6
        elif mlp_0_0_output in {3, 4, 22}:
            return position == 2
        elif mlp_0_0_output in {16, 5, 6}:
            return position == 23
        elif mlp_0_0_output in {7}:
            return position == 21
        elif mlp_0_0_output in {8}:
            return position == 16
        elif mlp_0_0_output in {9}:
            return position == 0
        elif mlp_0_0_output in {10}:
            return position == 28
        elif mlp_0_0_output in {11}:
            return position == 25
        elif mlp_0_0_output in {19, 12}:
            return position == 5
        elif mlp_0_0_output in {13, 23}:
            return position == 12
        elif mlp_0_0_output in {14}:
            return position == 9
        elif mlp_0_0_output in {15}:
            return position == 17
        elif mlp_0_0_output in {17}:
            return position == 11
        elif mlp_0_0_output in {18, 26}:
            return position == 14
        elif mlp_0_0_output in {20}:
            return position == 3
        elif mlp_0_0_output in {21}:
            return position == 26
        elif mlp_0_0_output in {24}:
            return position == 19
        elif mlp_0_0_output in {28}:
            return position == 13
        elif mlp_0_0_output in {29}:
            return position == 15

    attn_2_4_pattern = select_closest(positions, mlp_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_0_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 20
        elif q_position in {1}:
            return k_position == 21
        elif q_position in {2, 3}:
            return k_position == 4
        elif q_position in {19, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {6, 11, 12, 20, 27}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {8, 17}:
            return k_position == 6
        elif q_position in {9, 10}:
            return k_position == 13
        elif q_position in {13, 14, 15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 10
        elif q_position in {18, 21}:
            return k_position == 0
        elif q_position in {29, 22}:
            return k_position == 27
        elif q_position in {24, 23}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 1
        elif q_position in {26}:
            return k_position == 22
        elif q_position in {28}:
            return k_position == 2

    attn_2_5_pattern = select_closest(positions, positions, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_0_7_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
        if q_token in {"4", "0"}:
            return k_token == "2"
        elif q_token in {"2", "1", "<s>"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"</s>"}:
            return k_token == "</s>"

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_5_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_7_output, mlp_0_0_output):
        if attn_0_7_output in {"0"}:
            return mlp_0_0_output == 8
        elif attn_0_7_output in {"1"}:
            return mlp_0_0_output == 6
        elif attn_0_7_output in {"2"}:
            return mlp_0_0_output == 7
        elif attn_0_7_output in {"3"}:
            return mlp_0_0_output == 25
        elif attn_0_7_output in {"4"}:
            return mlp_0_0_output == 19
        elif attn_0_7_output in {"</s>"}:
            return mlp_0_0_output == 21
        elif attn_0_7_output in {"<s>"}:
            return mlp_0_0_output == 0

    attn_2_7_pattern = select_closest(mlp_0_0_outputs, attn_0_7_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_3_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, token):
        if position in {0, 4, 5, 6, 7, 19, 20, 23, 25, 26, 27, 28, 29}:
            return token == "2"
        elif position in {1, 9, 10, 13, 15, 16, 17, 18, 21, 22, 24}:
            return token == ""
        elif position in {8, 11, 2, 3}:
            return token == "<s>"
        elif position in {12, 14}:
            return token == "<pad>"

    num_attn_2_0_pattern = select(tokens, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, attn_1_0_output):
        if attn_1_1_output in {"4", "0"}:
            return attn_1_0_output == "1"
        elif attn_1_1_output in {"2", "1", "</s>"}:
            return attn_1_0_output == "0"
        elif attn_1_1_output in {"3"}:
            return attn_1_0_output == "2"
        elif attn_1_1_output in {"<s>"}:
            return attn_1_0_output == ""

    num_attn_2_1_pattern = select(attn_1_0_outputs, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_1_output, token):
        if attn_1_1_output in {"2", "1", "0"}:
            return token == ""
        elif attn_1_1_output in {"3"}:
            return token == "<s>"
        elif attn_1_1_output in {"<s>", "4", "</s>"}:
            return token == "1"

    num_attn_2_2_pattern = select(tokens, attn_1_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_1_output, attn_1_4_output):
        if attn_1_1_output in {"1", "4", "0"}:
            return attn_1_4_output == "3"
        elif attn_1_1_output in {"2"}:
            return attn_1_4_output == "2"
        elif attn_1_1_output in {"3", "</s>"}:
            return attn_1_4_output == "1"
        elif attn_1_1_output in {"<s>"}:
            return attn_1_4_output == ""

    num_attn_2_3_pattern = select(attn_1_4_outputs, attn_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(position, attn_1_4_output):
        if position in {
            0,
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            9,
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
            return attn_1_4_output == "0"
        elif position in {6}:
            return attn_1_4_output == "1"
        elif position in {10, 11, 12, 13, 14, 15, 16, 17, 18}:
            return attn_1_4_output == ""

    num_attn_2_4_pattern = select(attn_1_4_outputs, positions, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_1_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_1_output, attn_1_4_output):
        if attn_1_1_output in {"3", "0", "<s>"}:
            return attn_1_4_output == "0"
        elif attn_1_1_output in {"1", "4", "</s>"}:
            return attn_1_4_output == ""
        elif attn_1_1_output in {"2"}:
            return attn_1_4_output == "</s>"

    num_attn_2_5_pattern = select(attn_1_4_outputs, attn_1_1_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_2_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(position, attn_0_2_output):
        if position in {0, 1, 6, 7, 9, 10, 20, 21, 23, 26, 28, 29}:
            return attn_0_2_output == "1"
        elif position in {25, 2, 3, 4}:
            return attn_0_2_output == "2"
        elif position in {5, 8, 11, 12, 19, 22, 24, 27}:
            return attn_0_2_output == "0"
        elif position in {13}:
            return attn_0_2_output == "<s>"
        elif position in {14, 15, 16, 17, 18}:
            return attn_0_2_output == ""

    num_attn_2_6_pattern = select(attn_0_2_outputs, positions, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_2_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_1_output, attn_1_4_output):
        if attn_1_1_output in {"0", "</s>"}:
            return attn_1_4_output == "2"
        elif attn_1_1_output in {"2", "1", "<s>"}:
            return attn_1_4_output == "1"
        elif attn_1_1_output in {"3", "4"}:
            return attn_1_4_output == ""

    num_attn_2_7_pattern = select(attn_1_4_outputs, attn_1_1_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_3_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_1_3_output, attn_0_7_output):
        key = (mlp_1_3_output, attn_0_7_output)
        return 7

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_1_3_outputs, attn_0_7_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_2_output, attn_0_6_output):
        key = (attn_0_2_output, attn_0_6_output)
        if key in {
            ("0", "2"),
            ("0", "4"),
            ("1", "2"),
            ("1", "4"),
            ("2", "2"),
            ("2", "4"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "4"),
            ("4", "2"),
            ("4", "4"),
            ("</s>", "2"),
            ("</s>", "4"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 15
        return 13

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_6_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(position, attn_2_5_output):
        key = (position, attn_2_5_output)
        if key in {(2, "2"), (14, "</s>")}:
            return 23
        elif key in {
            (0, "0"),
            (0, "4"),
            (2, "4"),
            (4, "0"),
            (4, "2"),
            (4, "4"),
            (10, "1"),
            (10, "2"),
            (10, "4"),
            (11, "0"),
            (11, "1"),
            (11, "2"),
            (11, "4"),
            (11, "</s>"),
            (11, "<s>"),
            (14, "0"),
            (14, "3"),
            (14, "4"),
            (19, "0"),
            (19, "4"),
            (20, "0"),
            (20, "4"),
            (21, "0"),
            (21, "4"),
            (22, "0"),
            (22, "4"),
            (23, "0"),
            (23, "4"),
            (24, "0"),
            (24, "4"),
            (25, "0"),
            (25, "4"),
            (26, "0"),
            (26, "4"),
            (27, "0"),
            (27, "4"),
            (28, "0"),
            (28, "4"),
            (29, "0"),
            (29, "4"),
        }:
            return 12
        elif key in {
            (0, "3"),
            (0, "</s>"),
            (2, "0"),
            (2, "3"),
            (2, "</s>"),
            (3, "0"),
            (3, "3"),
            (3, "4"),
            (3, "</s>"),
            (4, "3"),
            (4, "</s>"),
            (10, "0"),
            (10, "3"),
            (10, "</s>"),
            (10, "<s>"),
            (11, "3"),
            (13, "0"),
            (13, "3"),
            (13, "4"),
            (13, "</s>"),
            (19, "3"),
            (19, "</s>"),
            (20, "3"),
            (20, "</s>"),
            (21, "3"),
            (21, "</s>"),
            (22, "3"),
            (22, "</s>"),
            (23, "3"),
            (23, "</s>"),
            (24, "3"),
            (24, "</s>"),
            (25, "3"),
            (25, "</s>"),
            (26, "3"),
            (26, "</s>"),
            (27, "3"),
            (27, "</s>"),
            (28, "3"),
            (28, "</s>"),
            (29, "3"),
            (29, "</s>"),
        }:
            return 5
        elif key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
        }:
            return 13
        return 22

    mlp_2_2_outputs = [mlp_2_2(k0, k1) for k0, k1 in zip(positions, attn_2_5_outputs)]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_1_3_output, attn_1_0_output):
        key = (num_mlp_1_3_output, attn_1_0_output)
        if key in {(7, "2"), (8, "2")}:
            return 25
        return 22

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_1_3_outputs, attn_1_0_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_2_3_output):
        key = (num_attn_1_2_output, num_attn_2_3_output)
        return 27

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output, num_attn_1_4_output):
        key = (num_attn_2_3_output, num_attn_1_4_output)
        return 28

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_0_6_output):
        key = num_attn_0_6_output
        if key in {0}:
            return 26
        return 18

    num_mlp_2_2_outputs = [num_mlp_2_2(k0) for k0 in num_attn_0_6_outputs]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_5_output, num_attn_2_7_output):
        key = (num_attn_1_5_output, num_attn_2_7_output)
        return 29

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_2_7_outputs)
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
            "0",
            "0",
            "2",
            "1",
            "2",
            "4",
            "1",
            "0",
            "4",
            "2",
            "4",
            "2",
            "4",
            "3",
            "</s>",
        ]
    )
)
