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
        "output/length/rasp/reverse/trainlength10/s1/reverse_weights.csv",
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
        if q_position in {0, 8}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2, 3}:
            return k_position == 0
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 10
        elif q_position in {18, 11}:
            return k_position == 17
        elif q_position in {17, 12}:
            return k_position == 18
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 11
        elif q_position in {19}:
            return k_position == 19

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 0
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {8, 6, 7}:
            return k_position == 1
        elif q_position in {9}:
            return k_position == 2
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {19, 11}:
            return k_position == 13
        elif q_position in {16, 12, 15}:
            return k_position == 11
        elif q_position in {18, 13}:
            return k_position == 19
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 16

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 3
        elif q_position in {8, 1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 0
        elif q_position in {9, 4, 7}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 1
        elif q_position in {16, 10}:
            return k_position == 12
        elif q_position in {19, 11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {13, 14}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 10
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {18}:
            return k_position == 16

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 9
        elif q_position in {4, 6, 7}:
            return k_position == 2
        elif q_position in {9, 5}:
            return k_position == 1
        elif q_position in {8}:
            return k_position == 5
        elif q_position in {10, 11}:
            return k_position == 16
        elif q_position in {16, 12}:
            return k_position == 13
        elif q_position in {13, 14}:
            return k_position == 17
        elif q_position in {19, 15}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 3, 4}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {9, 6}:
            return k_position == 2
        elif q_position in {7}:
            return k_position == 0
        elif q_position in {8}:
            return k_position == 1
        elif q_position in {10, 12, 15}:
            return k_position == 12
        elif q_position in {16, 11, 13}:
            return k_position == 17
        elif q_position in {18, 14}:
            return k_position == 10
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {19}:
            return k_position == 11

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 9}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {8, 2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 0
        elif q_position in {5}:
            return k_position == 1
        elif q_position in {6, 7}:
            return k_position == 2
        elif q_position in {10, 11, 15}:
            return k_position == 17
        elif q_position in {18, 12}:
            return k_position == 10
        elif q_position in {19, 13, 14}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 18

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8, 9}:
            return k_position == 1
        elif q_position in {10}:
            return k_position == 18
        elif q_position in {11}:
            return k_position == 11
        elif q_position in {12}:
            return k_position == 16
        elif q_position in {19, 13}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 14
        elif q_position in {17, 18}:
            return k_position == 12

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
            return k_position == 7
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {5}:
            return k_position == 0
        elif q_position in {8, 6}:
            return k_position == 1
        elif q_position in {9, 7}:
            return k_position == 2
        elif q_position in {10, 19}:
            return k_position == 15
        elif q_position in {11}:
            return k_position == 19
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {14}:
            return k_position == 14
        elif q_position in {15}:
            return k_position == 10
        elif q_position in {16, 18}:
            return k_position == 18
        elif q_position in {17}:
            return k_position == 12

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"1", "2", "3", "<s>", "</s>", "0"}:
            return position == 9
        elif token in {"4"}:
            return position == 8

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"1", "0"}:
            return position == 7
        elif token in {"4", "3", "2", "<s>"}:
            return position == 8
        elif token in {"</s>"}:
            return position == 10

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"4", "3", "2", "<s>", "</s>", "0"}:
            return position == 9
        elif token in {"1"}:
            return position == 8

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"0"}:
            return position == 8
        elif token in {"4", "1", "2", "<s>", "</s>"}:
            return position == 9
        elif token in {"3"}:
            return position == 10

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"0"}:
            return position == 5
        elif token in {"<s>", "1", "2", "</s>"}:
            return position == 9
        elif token in {"3"}:
            return position == 10
        elif token in {"4"}:
            return position == 8

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"4", "1", "2", "<s>", "</s>", "0"}:
            return position == 9
        elif token in {"3"}:
            return position == 8

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"4", "1", "<s>", "</s>", "0"}:
            return position == 9
        elif token in {"2"}:
            return position == 8
        elif token in {"3"}:
            return position == 12

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"4", "1", "2", "3", "<s>", "</s>", "0"}:
            return position == 9

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_6_output, attn_0_7_output):
        key = (attn_0_6_output, attn_0_7_output)
        if key in {
            ("0", "3"),
            ("1", "3"),
            ("2", "3"),
            ("3", "3"),
            ("4", "3"),
            ("<s>", "3"),
        }:
            return 15
        return 7

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_7_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_6_output, attn_0_7_output):
        key = (attn_0_6_output, attn_0_7_output)
        if key in {
            ("0", "4"),
            ("1", "4"),
            ("2", "4"),
            ("3", "4"),
            ("4", "4"),
            ("<s>", "4"),
        }:
            return 10
        elif key in {("</s>", "2")}:
            return 8
        return 11

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_7_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_2_output):
        key = (num_attn_0_0_output, num_attn_0_2_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
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
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
        }:
            return 7
        return 5

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_7_output):
        key = num_attn_0_7_output
        return 12

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_7_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 8, 3}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {4, 6}:
            return k_position == 5
        elif q_position in {9, 15, 5, 7}:
            return k_position == 6
        elif q_position in {10, 18, 14}:
            return k_position == 18
        elif q_position in {17, 11, 13}:
            return k_position == 15
        elif q_position in {12}:
            return k_position == 17
        elif q_position in {16}:
            return k_position == 10
        elif q_position in {19}:
            return k_position == 13

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_5_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 9}:
            return k_position == 9
        elif q_position in {8, 1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {4, 6}:
            return k_position == 8
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 1
        elif q_position in {10, 15}:
            return k_position == 11
        elif q_position in {11, 13}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {18, 14}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 10
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {19}:
            return k_position == 16

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_4_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {3, 4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {9, 7}:
            return k_position == 1
        elif q_position in {8}:
            return k_position == 2
        elif q_position in {10, 11}:
            return k_position == 12
        elif q_position in {17, 12, 13, 14}:
            return k_position == 15
        elif q_position in {18, 15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 10
        elif q_position in {19}:
            return k_position == 11

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_4_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 0
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {8, 9, 6, 7}:
            return k_position == 1
        elif q_position in {10}:
            return k_position == 18
        elif q_position in {16, 11}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 15
        elif q_position in {17, 19, 13}:
            return k_position == 16
        elif q_position in {18, 14}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 19

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 0
        elif q_position in {1, 6}:
            return k_position == 3
        elif q_position in {8, 2}:
            return k_position == 2
        elif q_position in {4, 7}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {16, 9, 13, 14}:
            return k_position == 9
        elif q_position in {10}:
            return k_position == 10
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18, 19}:
            return k_position == 12

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, token):
        if position in {0, 17}:
            return token == "2"
        elif position in {1, 2, 4}:
            return token == "3"
        elif position in {3, 5, 7, 10, 12, 13}:
            return token == "4"
        elif position in {6}:
            return token == "<s>"
        elif position in {8, 9}:
            return token == "</s>"
        elif position in {16, 19, 11, 15}:
            return token == ""
        elif position in {18, 14}:
            return token == "1"

    attn_1_5_pattern = select_closest(tokens, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {8, 3, 7}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5, 6}:
            return k_position == 8
        elif q_position in {9}:
            return k_position == 9
        elif q_position in {10, 15}:
            return k_position == 12
        elif q_position in {11, 12}:
            return k_position == 11
        elif q_position in {19, 13, 14}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 16
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {18}:
            return k_position == 14

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_2_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {8, 3, 4}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 8
        elif q_position in {9}:
            return k_position == 9
        elif q_position in {10, 11}:
            return k_position == 18
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {16, 14}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 15
        elif q_position in {19}:
            return k_position == 12

    attn_1_7_pattern = select_closest(positions, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_6_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_4_output, attn_0_1_output):
        if attn_0_4_output in {"1", "0"}:
            return attn_0_1_output == "<s>"
        elif attn_0_4_output in {"2"}:
            return attn_0_1_output == "2"
        elif attn_0_4_output in {"3"}:
            return attn_0_1_output == "3"
        elif attn_0_4_output in {"4"}:
            return attn_0_1_output == ""
        elif attn_0_4_output in {"</s>"}:
            return attn_0_1_output == "0"
        elif attn_0_4_output in {"<s>"}:
            return attn_0_1_output == "<pad>"

    num_attn_1_0_pattern = select(attn_0_1_outputs, attn_0_4_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_4_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"4", "0", "2"}:
            return k_attn_0_6_output == "1"
        elif q_attn_0_6_output in {"1"}:
            return k_attn_0_6_output == "<pad>"
        elif q_attn_0_6_output in {"3", "</s>"}:
            return k_attn_0_6_output == ""
        elif q_attn_0_6_output in {"<s>"}:
            return k_attn_0_6_output == "0"

    num_attn_1_1_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_7_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"0"}:
            return k_attn_0_6_output == "<pad>"
        elif q_attn_0_6_output in {"1", "3"}:
            return k_attn_0_6_output == "4"
        elif q_attn_0_6_output in {"2"}:
            return k_attn_0_6_output == "2"
        elif q_attn_0_6_output in {"4", "<s>", "</s>"}:
            return k_attn_0_6_output == ""

    num_attn_1_2_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_7_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"0", "2"}:
            return k_attn_0_6_output == "<pad>"
        elif q_attn_0_6_output in {"<s>", "1"}:
            return k_attn_0_6_output == ""
        elif q_attn_0_6_output in {"4", "3"}:
            return k_attn_0_6_output == "0"
        elif q_attn_0_6_output in {"</s>"}:
            return k_attn_0_6_output == "4"

    num_attn_1_3_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_7_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"4", "3", "<s>", "</s>", "0"}:
            return k_attn_0_6_output == ""
        elif q_attn_0_6_output in {"1", "2"}:
            return k_attn_0_6_output == "0"

    num_attn_1_4_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_7_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"0", "2"}:
            return k_attn_0_6_output == "3"
        elif q_attn_0_6_output in {"4", "1", "3", "<s>"}:
            return k_attn_0_6_output == ""
        elif q_attn_0_6_output in {"</s>"}:
            return k_attn_0_6_output == "1"

    num_attn_1_5_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"1", "2", "<s>", "</s>", "0"}:
            return k_attn_0_6_output == ""
        elif q_attn_0_6_output in {"4", "3"}:
            return k_attn_0_6_output == "2"

    num_attn_1_6_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_7_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"1", "2", "3", "<s>", "0"}:
            return k_attn_0_6_output == ""
        elif q_attn_0_6_output in {"4"}:
            return k_attn_0_6_output == "4"
        elif q_attn_0_6_output in {"</s>"}:
            return k_attn_0_6_output == "<pad>"

    num_attn_1_7_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_7_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_7_output, attn_0_4_output):
        key = (attn_0_7_output, attn_0_4_output)
        if key in {
            ("</s>", "0"),
            ("</s>", "2"),
            ("</s>", "</s>"),
            ("<s>", "0"),
            ("<s>", "</s>"),
        }:
            return 8
        elif key in {("1", "</s>"), ("3", "</s>"), ("4", "</s>")}:
            return 12
        elif key in {("2", "</s>")}:
            return 4
        elif key in {("0", "</s>")}:
            return 11
        elif key in {("</s>", "<s>")}:
            return 13
        return 7

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_4_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_7_output, attn_0_6_output):
        key = (attn_0_7_output, attn_0_6_output)
        if key in {
            ("0", "3"),
            ("0", "4"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "3"),
            ("1", "4"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("2", "4"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("3", "4"),
            ("3", "</s>"),
            ("4", "4"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "4"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 7
        elif key in {
            ("0", "1"),
            ("1", "1"),
            ("2", "1"),
            ("3", "1"),
            ("4", "1"),
            ("<s>", "1"),
        }:
            return 17
        return 0

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_6_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_6_output, num_attn_1_0_output):
        key = (num_attn_0_6_output, num_attn_1_0_output)
        return 17

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_6_output, num_attn_1_1_output):
        key = (num_attn_1_6_output, num_attn_1_1_output)
        return 0

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 5}:
            return k_position == 8
        elif q_position in {8, 3, 7}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {9, 10}:
            return k_position == 15
        elif q_position in {19, 17, 11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 18
        elif q_position in {13}:
            return k_position == 17
        elif q_position in {16, 14}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 13

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_6_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 1, 3, 8, 19}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {9, 11}:
            return k_position == 19
        elif q_position in {10}:
            return k_position == 18
        elif q_position in {16, 12}:
            return k_position == 10
        elif q_position in {13, 15}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 16
        elif q_position in {17}:
            return k_position == 11
        elif q_position in {18}:
            return k_position == 12

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, token):
        if position in {0, 2, 7}:
            return token == "</s>"
        elif position in {1, 3, 4, 5, 8}:
            return token == "<s>"
        elif position in {6}:
            return token == "3"
        elif position in {9}:
            return token == "4"
        elif position in {10, 11, 15}:
            return token == "1"
        elif position in {12, 13, 14, 16, 17, 18, 19}:
            return token == ""

    attn_2_2_pattern = select_closest(tokens, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(position, token):
        if position in {0, 9, 11, 12, 15, 17, 18}:
            return token == ""
        elif position in {1, 5, 7}:
            return token == "<s>"
        elif position in {8, 2, 3}:
            return token == "</s>"
        elif position in {16, 4, 13, 6}:
            return token == "0"
        elif position in {10, 19, 14}:
            return token == "2"

    attn_2_3_pattern = select_closest(tokens, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_7_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(position, token):
        if position in {0, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == "2"
        elif position in {1}:
            return token == "4"
        elif position in {9, 6}:
            return token == ""
        elif position in {8}:
            return token == "<pad>"

    attn_2_4_pattern = select_closest(tokens, positions, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_3_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 16
        elif q_position in {8, 1, 3}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7, 9, 16, 17, 18}:
            return k_position == 9
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13, 14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {19}:
            return k_position == 19

    attn_2_5_pattern = select_closest(positions, positions, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_3_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(position, token):
        if position in {0, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == "2"
        elif position in {1, 4}:
            return token == "<s>"
        elif position in {2}:
            return token == "</s>"
        elif position in {3}:
            return token == "0"
        elif position in {6}:
            return token == ""
        elif position in {7}:
            return token == "1"

    attn_2_6_pattern = select_closest(tokens, positions, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_2_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(position, token):
        if position in {0, 3, 10, 11, 12, 13, 15, 17, 18}:
            return token == "2"
        elif position in {1}:
            return token == "</s>"
        elif position in {8, 2}:
            return token == "1"
        elif position in {4}:
            return token == "<s>"
        elif position in {5, 6, 7, 9, 19}:
            return token == ""
        elif position in {16, 14}:
            return token == "3"

    attn_2_7_pattern = select_closest(tokens, positions, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"4", "0"}:
            return k_attn_0_6_output == "<pad>"
        elif q_attn_0_6_output in {"1", "</s>"}:
            return k_attn_0_6_output == ""
        elif q_attn_0_6_output in {"2"}:
            return k_attn_0_6_output == "4"
        elif q_attn_0_6_output in {"3"}:
            return k_attn_0_6_output == "3"
        elif q_attn_0_6_output in {"<s>"}:
            return k_attn_0_6_output == "</s>"

    num_attn_2_0_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_7_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_4_output, attn_0_1_output):
        if attn_0_4_output in {"1", "0", "</s>"}:
            return attn_0_1_output == ""
        elif attn_0_4_output in {"2"}:
            return attn_0_1_output == "3"
        elif attn_0_4_output in {"4", "3"}:
            return attn_0_1_output == "4"
        elif attn_0_4_output in {"<s>"}:
            return attn_0_1_output == "1"

    num_attn_2_1_pattern = select(attn_0_1_outputs, attn_0_4_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"0"}:
            return k_attn_0_6_output == "4"
        elif q_attn_0_6_output in {"<s>", "1", "3", "</s>"}:
            return k_attn_0_6_output == ""
        elif q_attn_0_6_output in {"4", "2"}:
            return k_attn_0_6_output == "2"

    num_attn_2_2_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_7_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"0"}:
            return k_attn_0_6_output == "0"
        elif q_attn_0_6_output in {"4", "1", "2", "<s>", "</s>"}:
            return k_attn_0_6_output == ""
        elif q_attn_0_6_output in {"3"}:
            return k_attn_0_6_output == "1"

    num_attn_2_3_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_7_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_3_output, attn_0_6_output):
        if attn_1_3_output in {"4", "0"}:
            return attn_0_6_output == "0"
        elif attn_1_3_output in {"1"}:
            return attn_0_6_output == "4"
        elif attn_1_3_output in {"<s>", "2"}:
            return attn_0_6_output == ""
        elif attn_1_3_output in {"3"}:
            return attn_0_6_output == "3"
        elif attn_1_3_output in {"</s>"}:
            return attn_0_6_output == "<pad>"

    num_attn_2_4_pattern = select(attn_0_6_outputs, attn_1_3_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_3_output, attn_0_6_output):
        if attn_1_3_output in {"0", "</s>"}:
            return attn_0_6_output == "<pad>"
        elif attn_1_3_output in {"1"}:
            return attn_0_6_output == "1"
        elif attn_1_3_output in {"4", "2", "<s>"}:
            return attn_0_6_output == ""
        elif attn_1_3_output in {"3"}:
            return attn_0_6_output == "2"

    num_attn_2_5_pattern = select(attn_0_6_outputs, attn_1_3_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_1_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"0"}:
            return k_attn_0_6_output == "2"
        elif q_attn_0_6_output in {"1"}:
            return k_attn_0_6_output == "1"
        elif q_attn_0_6_output in {"4", "3", "2", "<s>"}:
            return k_attn_0_6_output == ""
        elif q_attn_0_6_output in {"</s>"}:
            return k_attn_0_6_output == "</s>"

    num_attn_2_6_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_7_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_3_output, attn_0_7_output):
        if attn_1_3_output in {"0"}:
            return attn_0_7_output == "0"
        elif attn_1_3_output in {"<s>", "1", "2"}:
            return attn_0_7_output == ""
        elif attn_1_3_output in {"3", "</s>"}:
            return attn_0_7_output == "2"
        elif attn_1_3_output in {"4"}:
            return attn_0_7_output == "<s>"

    num_attn_2_7_pattern = select(attn_0_7_outputs, attn_1_3_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_1_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_0_output, attn_2_6_output):
        key = (attn_2_0_output, attn_2_6_output)
        if key in {
            ("0", "</s>"),
            ("1", "0"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("2", "</s>"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("4", "</s>"),
            ("</s>", "</s>"),
            ("<s>", "</s>"),
        }:
            return 5
        elif key in {("1", "1")}:
            return 11
        return 2

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_2_6_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_7_output, attn_0_3_output):
        key = (attn_0_7_output, attn_0_3_output)
        if key in {
            ("2", "</s>"),
            ("3", "</s>"),
            ("4", "</s>"),
            ("</s>", "</s>"),
            ("<s>", "</s>"),
        }:
            return 18
        elif key in {("0", "</s>")}:
            return 2
        elif key in {("1", "</s>")}:
            return 6
        return 8

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_6_output, num_attn_2_3_output):
        key = (num_attn_1_6_output, num_attn_2_3_output)
        return 0

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_7_output, num_attn_1_2_output):
        key = (num_attn_2_7_output, num_attn_1_2_output)
        return 13

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
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
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
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
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
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
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
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
