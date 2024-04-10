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
        "output/length/rasp/sort/trainlength20/s2/sort_weights.csv",
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
    def predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"4", "</s>", "1"}:
            return k_token == "4"
        elif q_token in {"<s>", "2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {19, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {21, 15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {26, 20}:
            return k_position == 27
        elif q_position in {24, 25, 22}:
            return k_position == 25
        elif q_position in {23}:
            return k_position == 29
        elif q_position in {27, 29}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 20

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 11, 12, 14, 15, 18}:
            return token == "1"
        elif position in {1, 2}:
            return token == "0"
        elif position in {3, 4, 5, 6, 7, 8, 9, 10, 16, 19}:
            return token == "4"
        elif position in {17, 13}:
            return token == "2"
        elif position in {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 4, 9, 11, 15, 16}:
            return token == "4"
        elif position in {1}:
            return token == "0"
        elif position in {19, 17, 2, 18}:
            return token == "1"
        elif position in {10, 3, 12, 14}:
            return token == "3"
        elif position in {5, 6, 7, 8, 13}:
            return token == "2"
        elif position in {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 7, 10, 11, 12, 14, 15}:
            return token == "1"
        elif position in {16, 1}:
            return token == "4"
        elif position in {17, 2, 4}:
            return token == "2"
        elif position in {3, 5, 6, 8, 9}:
            return token == "3"
        elif position in {13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {18, 19}:
            return token == "</s>"

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 17, 19, 15}:
            return token == "1"
        elif position in {1}:
            return token == "0"
        elif position in {2}:
            return token == "4"
        elif position in {3, 4, 5, 6, 7, 8, 9, 10, 11, 14}:
            return token == "3"
        elif position in {16, 12, 13}:
            return token == "2"
        elif position in {18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 14}:
            return k_position == 2
        elif q_position in {1, 18, 12, 7}:
            return k_position == 6
        elif q_position in {16, 2}:
            return k_position == 5
        elif q_position in {3, 4, 13}:
            return k_position == 8
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {8, 10, 11}:
            return k_position == 7
        elif q_position in {9, 15}:
            return k_position == 4
        elif q_position in {17}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 17
        elif q_position in {29, 27, 20, 21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 29
        elif q_position in {25, 23}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 12
        elif q_position in {26}:
            return k_position == 15
        elif q_position in {28}:
            return k_position == 22

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 9
        elif q_position in {2}:
            return k_position == 10
        elif q_position in {19, 3}:
            return k_position == 2
        elif q_position in {4, 13}:
            return k_position == 3
        elif q_position in {8, 17, 5, 7}:
            return k_position == 4
        elif q_position in {9, 10, 11, 6}:
            return k_position == 5
        elif q_position in {12}:
            return k_position == 8
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {16, 15}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 1
        elif q_position in {20, 23}:
            return k_position == 18
        elif q_position in {28, 21}:
            return k_position == 25
        elif q_position in {22}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 21
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {26, 29}:
            return k_position == 20
        elif q_position in {27}:
            return k_position == 17

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 12, 13, 14, 15, 16, 17}:
            return token == "1"
        elif position in {19, 1, 2, 18}:
            return token == "0"
        elif position in {3, 6, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {4, 5}:
            return token == "<pad>"
        elif position in {11, 7}:
            return token == "<s>"
        elif position in {8, 9, 10}:
            return token == "</s>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1}:
            return token == "</s>"
        elif position in {2, 3, 4, 6, 7, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == "0"
        elif position in {5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}:
            return token == ""

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 20}:
            return token == "1"
        elif position in {1}:
            return token == "<s>"
        elif position in {2, 3, 4, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == "0"
        elif position in {5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18}:
            return token == ""
        elif position in {12}:
            return token == "<pad>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1, 3, 4}:
            return token == "1"
        elif position in {2, 19, 6}:
            return token == "0"
        elif position in {
            5,
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
        elif position in {8}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 17, 18}:
            return token == "2"
        elif position in {1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {9, 11}:
            return token == "<s>"
        elif position in {10}:
            return token == "</s>"
        elif position in {12, 13, 14, 15, 16, 19}:
            return token == "3"
        elif position in {22}:
            return token == "0"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {20, 29, 7}:
            return k_position == 10
        elif q_position in {8, 19, 22, 26, 27, 28}:
            return k_position == 11
        elif q_position in {24, 9, 25, 23}:
            return k_position == 12
        elif q_position in {10, 11}:
            return k_position == 15
        elif q_position in {12}:
            return k_position == 16
        elif q_position in {13, 14}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 26
        elif q_position in {17}:
            return k_position == 20
        elif q_position in {18}:
            return k_position == 22
        elif q_position in {21}:
            return k_position == 13

    num_attn_0_5_pattern = select(positions, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 12, 14, 15, 16, 17}:
            return token == "0"
        elif position in {1, 2, 3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {8, 4}:
            return token == "<s>"
        elif position in {5, 6, 7}:
            return token == "</s>"
        elif position in {9, 10, 11, 18, 19}:
            return token == "1"
        elif position in {13}:
            return token == "2"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 7, 8, 9, 10, 11, 19}:
            return token == "0"
        elif position in {1, 2, 12, 13, 15, 16, 17, 18, 20, 22, 23, 25, 27, 28}:
            return token == ""
        elif position in {3, 4, 5, 6, 21, 24, 26, 29}:
            return token == "1"
        elif position in {14}:
            return token == "<s>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}:
            return 19
        return 28

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {1, 2, 3}:
            return 5
        elif key in {4, 6}:
            return 20
        elif key in {19}:
            return 3
        return 0

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_6_output, position):
        key = (attn_0_6_output, position)
        if key in {
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 6),
            ("0", 10),
            ("1", 1),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 6),
            ("1", 10),
            ("1", 21),
            ("1", 25),
            ("1", 27),
            ("1", 28),
            ("2", 1),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 6),
            ("2", 10),
            ("2", 20),
            ("2", 21),
            ("2", 22),
            ("2", 23),
            ("2", 25),
            ("2", 27),
            ("2", 28),
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("3", 6),
            ("3", 10),
            ("3", 21),
            ("3", 25),
            ("3", 27),
            ("3", 28),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 6),
            ("4", 10),
            ("</s>", 0),
            ("</s>", 1),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 5),
            ("</s>", 6),
            ("</s>", 10),
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
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 8
        elif key in {
            ("0", 8),
            ("0", 9),
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("1", 8),
            ("1", 9),
            ("1", 11),
            ("1", 12),
            ("1", 13),
            ("2", 8),
            ("2", 9),
            ("2", 11),
            ("2", 12),
            ("2", 13),
            ("3", 8),
            ("3", 9),
            ("3", 11),
            ("3", 12),
            ("3", 13),
            ("4", 8),
            ("4", 9),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("</s>", 7),
            ("</s>", 8),
            ("</s>", 9),
            ("</s>", 11),
            ("</s>", 12),
            ("</s>", 13),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 13),
        }:
            return 22
        elif key in {
            ("0", 7),
            ("0", 19),
            ("1", 7),
            ("2", 7),
            ("2", 19),
            ("3", 7),
            ("4", 7),
            ("4", 19),
            ("<s>", 7),
            ("<s>", 18),
            ("<s>", 19),
        }:
            return 4
        return 21

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_6_outputs, positions)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_6_output, attn_0_5_output):
        key = (attn_0_6_output, attn_0_5_output)
        return 4

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_5_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output, num_attn_0_5_output):
        key = (num_attn_0_3_output, num_attn_0_5_output)
        return 1

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_6_output, num_attn_0_1_output):
        key = (num_attn_0_6_output, num_attn_0_1_output)
        return 4

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        return 4

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_5_output):
        key = num_attn_0_5_output
        return 0

    num_mlp_0_3_outputs = [num_mlp_0_3(k0) for k0 in num_attn_0_5_outputs]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 27}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 3}:
            return k_position == 7
        elif q_position in {8, 4, 5, 6}:
            return k_position == 11
        elif q_position in {7}:
            return k_position == 16
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {10, 11, 12}:
            return k_position == 18
        elif q_position in {16, 13}:
            return k_position == 15
        elif q_position in {14, 20, 22, 25, 26}:
            return k_position == 19
        elif q_position in {24, 15}:
            return k_position == 14
        elif q_position in {17}:
            return k_position == 9
        elif q_position in {18, 28, 21, 23}:
            return k_position == 0
        elif q_position in {19}:
            return k_position == 5
        elif q_position in {29}:
            return k_position == 28

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_4_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {0, 16, 13, 25}:
            return token == "1"
        elif position in {1}:
            return token == "0"
        elif position in {2, 3, 4, 6, 14, 15, 28}:
            return token == "2"
        elif position in {5, 7, 8, 9, 10, 11, 12, 17, 18, 19}:
            return token == "4"
        elif position in {26, 20, 22}:
            return token == "3"
        elif position in {21, 23, 24, 27, 29}:
            return token == ""

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0}:
            return mlp_0_1_output == 24
        elif mlp_0_0_output in {1}:
            return mlp_0_1_output == 10
        elif mlp_0_0_output in {24, 2, 20, 14}:
            return mlp_0_1_output == 9
        elif mlp_0_0_output in {3}:
            return mlp_0_1_output == 20
        elif mlp_0_0_output in {4, 13}:
            return mlp_0_1_output == 1
        elif mlp_0_0_output in {5, 10, 15, 19, 26, 28}:
            return mlp_0_1_output == 5
        elif mlp_0_0_output in {6, 23}:
            return mlp_0_1_output == 13
        elif mlp_0_0_output in {7}:
            return mlp_0_1_output == 7
        elif mlp_0_0_output in {8}:
            return mlp_0_1_output == 28
        elif mlp_0_0_output in {9}:
            return mlp_0_1_output == 25
        elif mlp_0_0_output in {11}:
            return mlp_0_1_output == 18
        elif mlp_0_0_output in {12, 29}:
            return mlp_0_1_output == 3
        elif mlp_0_0_output in {16}:
            return mlp_0_1_output == 2
        elif mlp_0_0_output in {17, 25}:
            return mlp_0_1_output == 22
        elif mlp_0_0_output in {18}:
            return mlp_0_1_output == 14
        elif mlp_0_0_output in {27, 21}:
            return mlp_0_1_output == 6
        elif mlp_0_0_output in {22}:
            return mlp_0_1_output == 23

    attn_1_2_pattern = select_closest(mlp_0_1_outputs, mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_1_output, attn_0_5_output):
        if attn_0_1_output in {"0"}:
            return attn_0_5_output == "3"
        elif attn_0_1_output in {"2", "1"}:
            return attn_0_5_output == "0"
        elif attn_0_1_output in {"4", "3", "</s>", "<s>"}:
            return attn_0_5_output == ""

    attn_1_3_pattern = select_closest(attn_0_5_outputs, attn_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, mlp_0_2_output):
        if position in {0, 5, 6}:
            return mlp_0_2_output == 8
        elif position in {1, 14, 22, 23, 25}:
            return mlp_0_2_output == 1
        elif position in {16, 2, 18}:
            return mlp_0_2_output == 21
        elif position in {19, 3, 4}:
            return mlp_0_2_output == 3
        elif position in {7}:
            return mlp_0_2_output == 15
        elif position in {8, 20}:
            return mlp_0_2_output == 2
        elif position in {9, 10, 11}:
            return mlp_0_2_output == 4
        elif position in {17, 12, 15}:
            return mlp_0_2_output == 22
        elif position in {13}:
            return mlp_0_2_output == 5
        elif position in {21}:
            return mlp_0_2_output == 6
        elif position in {24}:
            return mlp_0_2_output == 28
        elif position in {26}:
            return mlp_0_2_output == 13
        elif position in {27, 28}:
            return mlp_0_2_output == 7
        elif position in {29}:
            return mlp_0_2_output == 9

    attn_1_4_pattern = select_closest(mlp_0_2_outputs, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_1_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 21}:
            return k_mlp_0_0_output == 5
        elif q_mlp_0_0_output in {1, 3, 7, 8, 16, 19}:
            return k_mlp_0_0_output == 28
        elif q_mlp_0_0_output in {2, 13}:
            return k_mlp_0_0_output == 16
        elif q_mlp_0_0_output in {4, 15, 17, 18, 20, 22, 25, 26, 27}:
            return k_mlp_0_0_output == 19
        elif q_mlp_0_0_output in {5}:
            return k_mlp_0_0_output == 7
        elif q_mlp_0_0_output in {9, 28, 6}:
            return k_mlp_0_0_output == 21
        elif q_mlp_0_0_output in {10, 11, 14, 23}:
            return k_mlp_0_0_output == 3
        elif q_mlp_0_0_output in {12}:
            return k_mlp_0_0_output == 2
        elif q_mlp_0_0_output in {24, 29}:
            return k_mlp_0_0_output == 24

    attn_1_5_pattern = select_closest(mlp_0_0_outputs, mlp_0_0_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_1_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, attn_0_4_output):
        if position in {0, 17, 6}:
            return attn_0_4_output == "4"
        elif position in {1, 12, 14}:
            return attn_0_4_output == "0"
        elif position in {8, 2, 13}:
            return attn_0_4_output == "1"
        elif position in {11, 9, 3, 7}:
            return attn_0_4_output == "3"
        elif position in {4, 18, 22, 24, 25, 27, 28, 29}:
            return attn_0_4_output == "</s>"
        elif position in {23, 5, 15}:
            return attn_0_4_output == ""
        elif position in {16, 10, 19, 20}:
            return attn_0_4_output == "2"
        elif position in {26, 21}:
            return attn_0_4_output == "<pad>"

    attn_1_6_pattern = select_closest(attn_0_4_outputs, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_0_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "0"
        elif attn_0_1_output in {"</s>", "1", "<s>"}:
            return token == ""
        elif attn_0_1_output in {"2"}:
            return token == "2"
        elif attn_0_1_output in {"3"}:
            return token == "3"
        elif attn_0_1_output in {"4"}:
            return token == "4"

    attn_1_7_pattern = select_closest(tokens, attn_0_1_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_3_output, token):
        if mlp_0_3_output in {
            0,
            1,
            2,
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
            return token == "2"
        elif mlp_0_3_output in {3}:
            return token == "0"
        elif mlp_0_3_output in {18}:
            return token == ""

    num_attn_1_0_pattern = select(tokens, mlp_0_3_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_6_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_0_output, attn_0_1_output):
        if mlp_0_0_output in {0, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 21}:
            return attn_0_1_output == ""
        elif mlp_0_0_output in {1}:
            return attn_0_1_output == "</s>"
        elif mlp_0_0_output in {2, 3}:
            return attn_0_1_output == "<s>"
        elif mlp_0_0_output in {24, 4}:
            return attn_0_1_output == "1"
        elif mlp_0_0_output in {5, 6, 7, 22, 23, 25, 26, 29}:
            return attn_0_1_output == "0"
        elif mlp_0_0_output in {16}:
            return attn_0_1_output == "3"
        elif mlp_0_0_output in {17}:
            return attn_0_1_output == "<pad>"
        elif mlp_0_0_output in {27, 20, 28}:
            return attn_0_1_output == "2"

    num_attn_1_1_pattern = select(attn_0_1_outputs, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, attn_0_3_output):
        if position in {0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 25}:
            return attn_0_3_output == ""
        elif position in {1, 2, 28}:
            return attn_0_3_output == "3"
        elif position in {3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 24, 26}:
            return attn_0_3_output == "2"
        elif position in {27, 29}:
            return attn_0_3_output == "0"

    num_attn_1_2_pattern = select(attn_0_3_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_6_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_3_output):
        if position in {0, 1, 2, 15, 16, 17, 18, 19, 21}:
            return attn_0_3_output == ""
        elif position in {3, 5, 6, 7, 8, 9, 23, 24}:
            return attn_0_3_output == "3"
        elif position in {10, 11, 4}:
            return attn_0_3_output == "2"
        elif position in {12, 13, 14}:
            return attn_0_3_output == "</s>"
        elif position in {26, 27, 20, 28}:
            return attn_0_3_output == "4"
        elif position in {22}:
            return attn_0_3_output == "0"
        elif position in {25}:
            return attn_0_3_output == "1"
        elif position in {29}:
            return attn_0_3_output == "<pad>"

    num_attn_1_3_pattern = select(attn_0_3_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_4_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 3, 5, 9, 12, 13, 14, 17, 18, 19, 21}:
            return token == ""
        elif mlp_0_0_output in {1}:
            return token == "3"
        elif mlp_0_0_output in {2, 4, 6, 7, 10, 11, 15, 20, 22, 23, 24, 25, 26, 29}:
            return token == "0"
        elif mlp_0_0_output in {8, 16, 27, 28}:
            return token == "1"

    num_attn_1_4_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_7_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, attn_0_1_output):
        if position in {0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21}:
            return attn_0_1_output == ""
        elif position in {1}:
            return attn_0_1_output == "3"
        elif position in {2}:
            return attn_0_1_output == "2"
        elif position in {3, 4, 5, 20, 22, 23, 24, 25, 26, 28, 29}:
            return attn_0_1_output == "0"
        elif position in {27, 6, 7}:
            return attn_0_1_output == "1"
        elif position in {8}:
            return attn_0_1_output == "</s>"

    num_attn_1_5_pattern = select(attn_0_1_outputs, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, attn_0_1_output):
        if position in {0, 12, 13, 23}:
            return attn_0_1_output == "0"
        elif position in {1, 3, 4, 5, 6, 7, 9, 10, 11, 19, 20, 22, 24, 25, 26, 27, 28}:
            return attn_0_1_output == "2"
        elif position in {2}:
            return attn_0_1_output == "<s>"
        elif position in {8, 29}:
            return attn_0_1_output == "1"
        elif position in {14}:
            return attn_0_1_output == "</s>"
        elif position in {15, 16, 17, 18, 21}:
            return attn_0_1_output == ""

    num_attn_1_6_pattern = select(attn_0_1_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_0_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(mlp_0_0_output, attn_0_2_output):
        if mlp_0_0_output in {0, 5, 6, 7, 17, 19, 21, 26}:
            return attn_0_2_output == ""
        elif mlp_0_0_output in {1}:
            return attn_0_2_output == "3"
        elif mlp_0_0_output in {
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
            20,
            22,
            23,
            24,
            25,
            27,
            28,
            29,
        }:
            return attn_0_2_output == "0"
        elif mlp_0_0_output in {9, 18}:
            return attn_0_2_output == "<pad>"

    num_attn_1_7_pattern = select(attn_0_2_outputs, mlp_0_0_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_0_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(mlp_0_0_output, attn_0_5_output):
        key = (mlp_0_0_output, attn_0_5_output)
        return 2

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, attn_0_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_6_output, position):
        key = (attn_1_6_output, position)
        if key in {
            ("0", 2),
            ("1", 2),
            ("1", 20),
            ("1", 24),
            ("1", 27),
            ("1", 28),
            ("2", 2),
            ("3", 2),
            ("4", 2),
            ("</s>", 2),
            ("<s>", 2),
        }:
            return 10
        elif key in {
            ("0", 1),
            ("0", 27),
            ("0", 28),
            ("1", 1),
            ("2", 1),
            ("2", 20),
            ("2", 24),
            ("2", 26),
            ("2", 27),
            ("2", 28),
            ("3", 1),
            ("4", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 4
        elif key in {
            ("0", 3),
            ("1", 0),
            ("1", 3),
            ("2", 0),
            ("2", 3),
            ("3", 3),
            ("4", 3),
            ("</s>", 3),
            ("<s>", 0),
            ("<s>", 3),
            ("<s>", 27),
        }:
            return 12
        elif key in {("1", 21), ("1", 22), ("1", 23), ("1", 25), ("1", 26), ("1", 29)}:
            return 15
        return 1

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_6_outputs, positions)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_0_output, mlp_0_0_output):
        key = (attn_1_0_output, mlp_0_0_output)
        if key in {
            ("4", 0),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 5),
            ("4", 9),
            ("4", 12),
            ("4", 13),
            ("4", 15),
            ("4", 17),
            ("4", 18),
            ("4", 21),
            ("4", 26),
            ("4", 29),
            ("</s>", 1),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 5),
            ("</s>", 8),
            ("</s>", 9),
            ("</s>", 10),
            ("</s>", 12),
            ("</s>", 13),
            ("</s>", 14),
            ("</s>", 15),
            ("</s>", 17),
            ("</s>", 18),
            ("</s>", 20),
            ("</s>", 21),
            ("</s>", 22),
            ("</s>", 23),
            ("</s>", 24),
            ("</s>", 25),
            ("</s>", 26),
            ("</s>", 29),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 12),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 15),
            ("<s>", 17),
            ("<s>", 18),
            ("<s>", 21),
            ("<s>", 22),
            ("<s>", 23),
            ("<s>", 24),
            ("<s>", 25),
            ("<s>", 26),
            ("<s>", 29),
        }:
            return 23
        elif key in {
            ("0", 19),
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
            ("1", 21),
            ("1", 22),
            ("1", 23),
            ("1", 24),
            ("1", 25),
            ("1", 26),
            ("1", 27),
            ("1", 28),
            ("1", 29),
            ("2", 19),
            ("3", 0),
            ("3", 1),
            ("3", 2),
            ("3", 3),
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
            ("3", 20),
            ("3", 21),
            ("3", 22),
            ("3", 23),
            ("3", 24),
            ("3", 25),
            ("3", 26),
            ("3", 27),
            ("3", 28),
            ("3", 29),
            ("4", 6),
            ("4", 7),
            ("4", 19),
            ("</s>", 0),
            ("</s>", 2),
            ("</s>", 6),
            ("</s>", 7),
            ("</s>", 19),
            ("<s>", 0),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 19),
        }:
            return 28
        return 12

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_0_outputs, mlp_0_0_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(mlp_0_2_output, mlp_0_3_output):
        key = (mlp_0_2_output, mlp_0_3_output)
        return 4

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, mlp_0_3_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_5_output, num_attn_1_4_output):
        key = (num_attn_1_5_output, num_attn_1_4_output)
        return 21

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_1_1_output):
        key = (num_attn_1_2_output, num_attn_1_1_output)
        return 4

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_4_output):
        key = num_attn_1_4_output
        return 6

    num_mlp_1_2_outputs = [num_mlp_1_2(k0) for k0 in num_attn_1_4_outputs]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_2_output, num_attn_1_1_output):
        key = (num_attn_1_2_output, num_attn_1_1_output)
        return 12

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_1_4_output, mlp_0_2_output):
        if attn_1_4_output in {"0"}:
            return mlp_0_2_output == 19
        elif attn_1_4_output in {"4", "3", "1"}:
            return mlp_0_2_output == 8
        elif attn_1_4_output in {"2"}:
            return mlp_0_2_output == 26
        elif attn_1_4_output in {"</s>", "<s>"}:
            return mlp_0_2_output == 6

    attn_2_0_pattern = select_closest(mlp_0_2_outputs, attn_1_4_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_7_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_0_0_output, attn_1_3_output):
        if mlp_0_0_output in {0, 28}:
            return attn_1_3_output == "</s>"
        elif mlp_0_0_output in {
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            12,
            13,
            14,
            15,
            17,
            19,
            21,
            22,
            23,
            29,
        }:
            return attn_1_3_output == ""
        elif mlp_0_0_output in {4, 10, 18, 20, 24, 25, 26, 27}:
            return attn_1_3_output == "2"
        elif mlp_0_0_output in {11}:
            return attn_1_3_output == "3"
        elif mlp_0_0_output in {16}:
            return attn_1_3_output == "1"

    attn_2_1_pattern = select_closest(attn_1_3_outputs, mlp_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_1_6_output, mlp_0_0_output):
        if attn_1_6_output in {"0", "2"}:
            return mlp_0_0_output == 28
        elif attn_1_6_output in {"1"}:
            return mlp_0_0_output == 22
        elif attn_1_6_output in {"3"}:
            return mlp_0_0_output == 12
        elif attn_1_6_output in {"4"}:
            return mlp_0_0_output == 6
        elif attn_1_6_output in {"</s>"}:
            return mlp_0_0_output == 5
        elif attn_1_6_output in {"<s>"}:
            return mlp_0_0_output == 7

    attn_2_2_pattern = select_closest(mlp_0_0_outputs, attn_1_6_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_7_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"0", "1"}:
            return k_token == "4"
        elif q_token in {"4", "2"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "2"
        elif q_token in {"</s>"}:
            return k_token == "</s>"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_7_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_3_output, token):
        if attn_0_3_output in {"0", "3", "1", "<s>"}:
            return token == ""
        elif attn_0_3_output in {"</s>", "2"}:
            return token == "<s>"
        elif attn_0_3_output in {"4"}:
            return token == "1"

    attn_2_4_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_4_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(position, token):
        if position in {0, 24}:
            return token == "<pad>"
        elif position in {1, 2, 5, 22}:
            return token == "1"
        elif position in {3, 6, 7}:
            return token == "</s>"
        elif position in {4, 9, 10, 11, 12, 13, 14, 15, 17, 18}:
            return token == "4"
        elif position in {8, 16}:
            return token == "0"
        elif position in {19, 20, 23, 25, 26}:
            return token == ""
        elif position in {29, 27, 28, 21}:
            return token == "2"

    attn_2_5_pattern = select_closest(tokens, positions, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, tokens)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 28}:
            return position == 6
        elif mlp_0_1_output in {1, 2, 11, 16, 20}:
            return position == 3
        elif mlp_0_1_output in {3}:
            return position == 22
        elif mlp_0_1_output in {4, 29}:
            return position == 26
        elif mlp_0_1_output in {19, 5}:
            return position == 4
        elif mlp_0_1_output in {21, 6}:
            return position == 0
        elif mlp_0_1_output in {10, 14, 7}:
            return position == 8
        elif mlp_0_1_output in {8, 24}:
            return position == 11
        elif mlp_0_1_output in {9, 12, 25}:
            return position == 7
        elif mlp_0_1_output in {17, 18, 13}:
            return position == 15
        elif mlp_0_1_output in {15}:
            return position == 13
        elif mlp_0_1_output in {22}:
            return position == 21
        elif mlp_0_1_output in {23}:
            return position == 23
        elif mlp_0_1_output in {26}:
            return position == 5
        elif mlp_0_1_output in {27}:
            return position == 16

    attn_2_6_pattern = select_closest(positions, mlp_0_1_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_7_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "1"
        elif q_token in {"4", "1"}:
            return k_token == "2"
        elif q_token in {"3", "2"}:
            return k_token == "4"
        elif q_token in {"</s>"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_7_pattern = select_closest(tokens, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_0_output, position):
        if attn_1_0_output in {"0", "</s>", "<s>"}:
            return position == 13
        elif attn_1_0_output in {"3", "1"}:
            return position == 0
        elif attn_1_0_output in {"2"}:
            return position == 16
        elif attn_1_0_output in {"4"}:
            return position == 12

    num_attn_2_0_pattern = select(positions, attn_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_0_2_output):
        if position in {0}:
            return attn_0_2_output == "<pad>"
        elif position in {1, 13}:
            return attn_0_2_output == "</s>"
        elif position in {2, 7, 8, 9, 20, 24, 29}:
            return attn_0_2_output == "1"
        elif position in {3, 4, 6, 12, 22, 23, 25, 26, 27}:
            return attn_0_2_output == "2"
        elif position in {28, 5}:
            return attn_0_2_output == "3"
        elif position in {10}:
            return attn_0_2_output == "0"
        elif position in {11}:
            return attn_0_2_output == "<s>"
        elif position in {14, 15, 16, 17, 18, 19, 21}:
            return attn_0_2_output == ""

    num_attn_2_1_pattern = select(attn_0_2_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_1_0_output, mlp_1_3_output):
        if mlp_1_0_output in {0, 18}:
            return mlp_1_3_output == 24
        elif mlp_1_0_output in {1}:
            return mlp_1_3_output == 28
        elif mlp_1_0_output in {2, 4, 23, 15}:
            return mlp_1_3_output == 22
        elif mlp_1_0_output in {10, 3}:
            return mlp_1_3_output == 8
        elif mlp_1_0_output in {5}:
            return mlp_1_3_output == 7
        elif mlp_1_0_output in {6}:
            return mlp_1_3_output == 19
        elif mlp_1_0_output in {9, 7}:
            return mlp_1_3_output == 9
        elif mlp_1_0_output in {8}:
            return mlp_1_3_output == 2
        elif mlp_1_0_output in {27, 11}:
            return mlp_1_3_output == 23
        elif mlp_1_0_output in {12}:
            return mlp_1_3_output == 1
        elif mlp_1_0_output in {20, 13}:
            return mlp_1_3_output == 20
        elif mlp_1_0_output in {14}:
            return mlp_1_3_output == 29
        elif mlp_1_0_output in {16, 21}:
            return mlp_1_3_output == 6
        elif mlp_1_0_output in {17}:
            return mlp_1_3_output == 10
        elif mlp_1_0_output in {26, 19}:
            return mlp_1_3_output == 21
        elif mlp_1_0_output in {22}:
            return mlp_1_3_output == 27
        elif mlp_1_0_output in {24}:
            return mlp_1_3_output == 15
        elif mlp_1_0_output in {25, 29}:
            return mlp_1_3_output == 0
        elif mlp_1_0_output in {28}:
            return mlp_1_3_output == 26

    num_attn_2_2_pattern = select(mlp_1_3_outputs, mlp_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_6_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, attn_1_7_output):
        if position in {
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
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
            return attn_1_7_output == "0"
        elif position in {1, 10}:
            return attn_1_7_output == "</s>"
        elif position in {11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return attn_1_7_output == ""

    num_attn_2_3_pattern = select(attn_1_7_outputs, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_6_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_0_output, attn_1_6_output):
        if attn_1_0_output in {"0"}:
            return attn_1_6_output == "0"
        elif attn_1_0_output in {"1"}:
            return attn_1_6_output == "</s>"
        elif attn_1_0_output in {"3", "2"}:
            return attn_1_6_output == ""
        elif attn_1_0_output in {"4", "</s>", "<s>"}:
            return attn_1_6_output == "1"

    num_attn_2_4_pattern = select(attn_1_6_outputs, attn_1_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_7_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_0_output, attn_0_7_output):
        if attn_1_0_output in {"0", "3", "2"}:
            return attn_0_7_output == "1"
        elif attn_1_0_output in {"1"}:
            return attn_0_7_output == "0"
        elif attn_1_0_output in {"4", "</s>", "<s>"}:
            return attn_0_7_output == "2"

    num_attn_2_5_pattern = select(attn_0_7_outputs, attn_1_0_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_0_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_0_output, attn_1_1_output):
        if attn_1_0_output in {"0", "3", "2"}:
            return attn_1_1_output == "2"
        elif attn_1_0_output in {"</s>", "1", "<s>"}:
            return attn_1_1_output == "1"
        elif attn_1_0_output in {"4"}:
            return attn_1_1_output == ""

    num_attn_2_6_pattern = select(attn_1_1_outputs, attn_1_0_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_6_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_0_output, num_mlp_1_1_output):
        if attn_1_0_output in {"0"}:
            return num_mlp_1_1_output == 2
        elif attn_1_0_output in {"1"}:
            return num_mlp_1_1_output == 28
        elif attn_1_0_output in {"2"}:
            return num_mlp_1_1_output == 6
        elif attn_1_0_output in {"3"}:
            return num_mlp_1_1_output == 7
        elif attn_1_0_output in {"4"}:
            return num_mlp_1_1_output == 0
        elif attn_1_0_output in {"</s>"}:
            return num_mlp_1_1_output == 3
        elif attn_1_0_output in {"<s>"}:
            return num_mlp_1_1_output == 4

    num_attn_2_7_pattern = select(
        num_mlp_1_1_outputs, attn_1_0_outputs, num_predicate_2_7
    )
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_4_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_7_output, attn_0_3_output):
        key = (attn_0_7_output, attn_0_3_output)
        return 15

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_0_3_output, mlp_1_1_output):
        key = (num_mlp_0_3_output, mlp_1_1_output)
        return 27

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_0_3_outputs, mlp_1_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(num_mlp_0_3_output):
        key = num_mlp_0_3_output
        return 29

    mlp_2_2_outputs = [mlp_2_2(k0) for k0 in num_mlp_0_3_outputs]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_2_2_output, attn_2_7_output):
        key = (attn_2_2_output, attn_2_7_output)
        if key in {("0", "1"), ("1", "1"), ("2", "1"), ("4", "1"), ("<s>", "1")}:
            return 26
        return 16

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_7_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output):
        key = num_attn_1_0_output
        return 11

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_1_0_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_5_output, num_attn_0_5_output):
        key = (num_attn_1_5_output, num_attn_0_5_output)
        if key in {(0, 0)}:
            return 25
        return 20

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_7_output, num_attn_2_2_output):
        key = (num_attn_1_7_output, num_attn_2_2_output)
        return 20

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_0_output, num_attn_1_7_output):
        key = (num_attn_1_0_output, num_attn_1_7_output)
        if key in {
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
            (14, 51),
            (14, 52),
            (14, 53),
            (14, 54),
            (14, 55),
            (14, 56),
            (14, 57),
            (14, 58),
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
            (15, 55),
            (15, 56),
            (15, 57),
            (15, 58),
            (15, 59),
            (15, 60),
            (15, 61),
            (15, 62),
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
            (16, 58),
            (16, 59),
            (16, 60),
            (16, 61),
            (16, 62),
            (16, 63),
            (16, 64),
            (16, 65),
            (16, 66),
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
            (17, 62),
            (17, 63),
            (17, 64),
            (17, 65),
            (17, 66),
            (17, 67),
            (17, 68),
            (17, 69),
            (17, 70),
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
            (18, 65),
            (18, 66),
            (18, 67),
            (18, 68),
            (18, 69),
            (18, 70),
            (18, 71),
            (18, 72),
            (18, 73),
            (18, 74),
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
            (19, 69),
            (19, 70),
            (19, 71),
            (19, 72),
            (19, 73),
            (19, 74),
            (19, 75),
            (19, 76),
            (19, 77),
            (19, 78),
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
            (20, 72),
            (20, 73),
            (20, 74),
            (20, 75),
            (20, 76),
            (20, 77),
            (20, 78),
            (20, 79),
            (20, 80),
            (20, 81),
            (20, 82),
            (20, 83),
            (20, 84),
            (20, 85),
            (20, 86),
            (20, 87),
            (20, 88),
            (20, 89),
            (21, 76),
            (21, 77),
            (21, 78),
            (21, 79),
            (21, 80),
            (21, 81),
            (21, 82),
            (21, 83),
            (21, 84),
            (21, 85),
            (21, 86),
            (21, 87),
            (21, 88),
            (21, 89),
            (22, 79),
            (22, 80),
            (22, 81),
            (22, 82),
            (22, 83),
            (22, 84),
            (22, 85),
            (22, 86),
            (22, 87),
            (22, 88),
            (22, 89),
            (23, 83),
            (23, 84),
            (23, 85),
            (23, 86),
            (23, 87),
            (23, 88),
            (23, 89),
            (24, 86),
            (24, 87),
            (24, 88),
            (24, 89),
        }:
            return 1
        return 26

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_7_outputs)
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


print(run(["<s>", "0", "3", "2", "3", "0", "2", "1", "3", "2", "</s>"]))
