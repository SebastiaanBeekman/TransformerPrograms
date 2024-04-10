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
        "output/length/rasp/double_hist/trainlength10/s4/double_hist_weights.csv",
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
        if q_position in {0, 1, 6, 15, 18}:
            return k_position == 6
        elif q_position in {2, 3, 4, 5}:
            return k_position == 7
        elif q_position in {8, 9, 7}:
            return k_position == 2
        elif q_position in {10, 12, 13, 14, 16, 17, 19}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 8

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 5, 6, 12, 18}:
            return k_position == 6
        elif q_position in {1, 4}:
            return k_position == 4
        elif q_position in {2, 7}:
            return k_position == 7
        elif q_position in {3, 10, 11, 13, 14, 15, 16, 17, 19}:
            return k_position == 5
        elif q_position in {8, 9}:
            return k_position == 2

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 3, 13, 15, 18}:
            return k_position == 5
        elif q_position in {1, 7}:
            return k_position == 9
        elif q_position in {2, 5}:
            return k_position == 7
        elif q_position in {4, 12, 6}:
            return k_position == 6
        elif q_position in {8, 9}:
            return k_position == 2
        elif q_position in {16, 10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 11
        elif q_position in {19}:
            return k_position == 16

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"4", "1", "2"}:
            return k_token == "5"
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"5"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 13
        elif q_position in {1, 2, 3, 4, 5, 7}:
            return k_position == 6
        elif q_position in {9, 6, 17}:
            return k_position == 5
        elif q_position in {8, 15}:
            return k_position == 2
        elif q_position in {10, 18}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 7
        elif q_position in {13, 14}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 9
        elif q_position in {19}:
            return k_position == 11

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"5", "0", "<s>", "1", "4", "3", "2"}:
            return k_token == ""

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 19
        elif q_position in {1, 2, 3}:
            return k_position == 4
        elif q_position in {4, 13}:
            return k_position == 13
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {10, 7}:
            return k_position == 10
        elif q_position in {8, 9, 11, 17}:
            return k_position == 11
        elif q_position in {12}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {18, 15}:
            return k_position == 9
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 19
        elif q_position in {1, 2, 3, 4, 14, 17}:
            return k_position == 6
        elif q_position in {8, 11, 5}:
            return k_position == 17
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 4
        elif q_position in {12}:
            return k_position == 3
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 2
        elif q_position in {16, 18}:
            return k_position == 1
        elif q_position in {19}:
            return k_position == 5

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output):
        key = attn_0_3_output
        if key in {""}:
            return 15
        elif key in {"", "2", "4", "5"}:
            return 0
        return 12

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in attn_0_3_outputs]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, token):
        key = (attn_0_3_output, token)
        return 12

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 17

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 0

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, mlp_0_0_output):
        if token in {"5", "0", "1", "4", "3", "2"}:
            return mlp_0_0_output == 15
        elif token in {"<s>"}:
            return mlp_0_0_output == 13

    attn_1_0_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_0_output, position):
        if attn_0_0_output in {0}:
            return position == 14
        elif attn_0_0_output in {1}:
            return position == 4
        elif attn_0_0_output in {2}:
            return position == 8
        elif attn_0_0_output in {3, 6, 7, 8, 9, 12, 13, 16, 17, 18, 19}:
            return position == 9
        elif attn_0_0_output in {4}:
            return position == 7
        elif attn_0_0_output in {5}:
            return position == 6
        elif attn_0_0_output in {10}:
            return position == 17
        elif attn_0_0_output in {11}:
            return position == 10
        elif attn_0_0_output in {14}:
            return position == 11
        elif attn_0_0_output in {15}:
            return position == 5

    attn_1_1_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_0_output, position):
        if attn_0_0_output in {0, 1, 5, 13, 14, 15, 16, 17}:
            return position == 6
        elif attn_0_0_output in {2, 8, 9, 10, 11, 12, 18}:
            return position == 9
        elif attn_0_0_output in {3}:
            return position == 4
        elif attn_0_0_output in {4}:
            return position == 7
        elif attn_0_0_output in {6, 7}:
            return position == 8
        elif attn_0_0_output in {19}:
            return position == 10

    attn_1_2_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_0_output, position):
        if attn_0_0_output in {0, 3, 5}:
            return position == 6
        elif attn_0_0_output in {1, 2, 6, 11, 15, 16, 19}:
            return position == 7
        elif attn_0_0_output in {4}:
            return position == 8
        elif attn_0_0_output in {7}:
            return position == 2
        elif attn_0_0_output in {8, 10}:
            return position == 14
        elif attn_0_0_output in {9, 17}:
            return position == 9
        elif attn_0_0_output in {12}:
            return position == 5
        elif attn_0_0_output in {13}:
            return position == 12
        elif attn_0_0_output in {14}:
            return position == 10
        elif attn_0_0_output in {18}:
            return position == 15

    attn_1_3_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0, 12, 5}:
            return mlp_0_1_output == 11
        elif mlp_0_0_output in {16, 1, 3, 6}:
            return mlp_0_1_output == 15
        elif mlp_0_0_output in {2}:
            return mlp_0_1_output == 6
        elif mlp_0_0_output in {17, 4}:
            return mlp_0_1_output == 2
        elif mlp_0_0_output in {9, 7}:
            return mlp_0_1_output == 8
        elif mlp_0_0_output in {8, 10, 18}:
            return mlp_0_1_output == 13
        elif mlp_0_0_output in {11}:
            return mlp_0_1_output == 12
        elif mlp_0_0_output in {13}:
            return mlp_0_1_output == 9
        elif mlp_0_0_output in {14, 15}:
            return mlp_0_1_output == 10
        elif mlp_0_0_output in {19}:
            return mlp_0_1_output == 16

    num_attn_1_0_pattern = select(mlp_0_1_outputs, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    num_attn_1_1_pattern = select(tokens, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"<s>", "2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"

    num_attn_1_2_pattern = select(tokens, tokens, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0"}:
            return k_attn_0_3_output == "<pad>"
        elif q_attn_0_3_output in {"5", "<s>", "1", "4", "3", "2"}:
            return k_attn_0_3_output == ""

    num_attn_1_3_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, attn_1_1_output):
        key = (attn_1_2_output, attn_1_1_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
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
            (1, 0),
            (1, 1),
            (1, 3),
            (1, 4),
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
            (3, 0),
            (3, 1),
            (3, 3),
            (3, 4),
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
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
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
            (10, 0),
            (10, 1),
            (10, 3),
            (10, 4),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (11, 0),
            (11, 1),
            (11, 3),
            (11, 4),
            (11, 10),
            (11, 11),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
            (11, 16),
            (11, 17),
            (11, 18),
            (11, 19),
            (12, 0),
            (12, 1),
            (12, 3),
            (12, 4),
            (12, 10),
            (12, 11),
            (12, 12),
            (12, 13),
            (12, 14),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 18),
            (12, 19),
            (13, 0),
            (13, 1),
            (13, 3),
            (13, 4),
            (13, 10),
            (13, 11),
            (13, 15),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (14, 0),
            (14, 1),
            (14, 3),
            (14, 4),
            (14, 10),
            (14, 11),
            (14, 15),
            (14, 16),
            (14, 17),
            (14, 18),
            (14, 19),
            (15, 0),
            (15, 1),
            (15, 3),
            (15, 4),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 14),
            (15, 15),
            (15, 16),
            (15, 17),
            (15, 18),
            (15, 19),
            (16, 0),
            (16, 1),
            (16, 3),
            (16, 4),
            (16, 10),
            (16, 11),
            (16, 15),
            (16, 16),
            (16, 17),
            (16, 18),
            (16, 19),
            (17, 0),
            (17, 1),
            (17, 3),
            (17, 4),
            (17, 10),
            (17, 11),
            (17, 12),
            (17, 13),
            (17, 14),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 18),
            (17, 19),
            (18, 0),
            (18, 1),
            (18, 3),
            (18, 4),
            (18, 10),
            (18, 11),
            (18, 12),
            (18, 13),
            (18, 14),
            (18, 15),
            (18, 16),
            (18, 17),
            (18, 18),
            (18, 19),
            (19, 0),
            (19, 1),
            (19, 3),
            (19, 4),
            (19, 10),
            (19, 11),
            (19, 12),
            (19, 13),
            (19, 14),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 18),
            (19, 19),
        }:
            return 0
        return 12

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_1_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(mlp_0_1_output, attn_1_3_output):
        key = (mlp_0_1_output, attn_1_3_output)
        if key in {
            (0, 0),
            (0, 3),
            (0, 4),
            (0, 12),
            (1, 0),
            (1, 3),
            (1, 4),
            (1, 12),
            (2, 0),
            (2, 4),
            (3, 0),
            (3, 4),
            (3, 12),
            (4, 0),
            (4, 4),
            (4, 12),
            (5, 0),
            (5, 3),
            (5, 4),
            (5, 12),
            (9, 0),
            (12, 0),
            (12, 4),
            (17, 0),
            (17, 4),
        }:
            return 12
        elif key in {(0, 1), (0, 17), (3, 3), (4, 3), (4, 8), (12, 12)}:
            return 19
        return 18

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_1_output, num_attn_0_0_output):
        key = (num_attn_1_1_output, num_attn_0_0_output)
        if key in {
            (2, 0),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (4, 2),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
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
            (9, 7),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (12, 9),
            (12, 10),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (13, 10),
            (13, 11),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 7),
            (14, 8),
            (14, 9),
            (14, 10),
            (14, 11),
            (14, 12),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 10),
            (16, 11),
            (16, 12),
            (16, 13),
            (16, 14),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 9),
            (17, 10),
            (17, 11),
            (17, 12),
            (17, 13),
            (17, 14),
            (17, 15),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (18, 9),
            (18, 10),
            (18, 11),
            (18, 12),
            (18, 13),
            (18, 14),
            (18, 15),
            (18, 16),
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
            (19, 10),
            (19, 11),
            (19, 12),
            (19, 13),
            (19, 14),
            (19, 15),
            (19, 16),
            (19, 17),
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
            (20, 10),
            (20, 11),
            (20, 12),
            (20, 13),
            (20, 14),
            (20, 15),
            (20, 16),
            (20, 17),
            (20, 18),
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
            (21, 11),
            (21, 12),
            (21, 13),
            (21, 14),
            (21, 15),
            (21, 16),
            (21, 17),
            (21, 18),
            (21, 19),
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
            (22, 12),
            (22, 13),
            (22, 14),
            (22, 15),
            (22, 16),
            (22, 17),
            (22, 18),
            (22, 19),
            (22, 20),
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
            (23, 12),
            (23, 13),
            (23, 14),
            (23, 15),
            (23, 16),
            (23, 17),
            (23, 18),
            (23, 19),
            (23, 20),
            (23, 21),
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
            (24, 13),
            (24, 14),
            (24, 15),
            (24, 16),
            (24, 17),
            (24, 18),
            (24, 19),
            (24, 20),
            (24, 21),
            (24, 22),
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
            (25, 13),
            (25, 14),
            (25, 15),
            (25, 16),
            (25, 17),
            (25, 18),
            (25, 19),
            (25, 20),
            (25, 21),
            (25, 22),
            (25, 23),
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
            (26, 16),
            (26, 17),
            (26, 18),
            (26, 19),
            (26, 20),
            (26, 21),
            (26, 22),
            (26, 23),
            (26, 24),
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
            (27, 14),
            (27, 15),
            (27, 16),
            (27, 17),
            (27, 18),
            (27, 19),
            (27, 20),
            (27, 21),
            (27, 22),
            (27, 23),
            (27, 24),
            (27, 25),
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
            (28, 15),
            (28, 16),
            (28, 17),
            (28, 18),
            (28, 19),
            (28, 20),
            (28, 21),
            (28, 22),
            (28, 23),
            (28, 24),
            (28, 25),
            (28, 26),
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
            (29, 15),
            (29, 16),
            (29, 17),
            (29, 18),
            (29, 19),
            (29, 20),
            (29, 21),
            (29, 22),
            (29, 23),
            (29, 24),
            (29, 25),
            (29, 26),
            (29, 27),
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
            (30, 16),
            (30, 17),
            (30, 18),
            (30, 19),
            (30, 20),
            (30, 21),
            (30, 22),
            (30, 23),
            (30, 24),
            (30, 25),
            (30, 26),
            (30, 27),
            (30, 28),
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
            (31, 17),
            (31, 18),
            (31, 19),
            (31, 20),
            (31, 21),
            (31, 22),
            (31, 23),
            (31, 24),
            (31, 25),
            (31, 26),
            (31, 27),
            (31, 28),
            (31, 29),
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
            (32, 17),
            (32, 18),
            (32, 19),
            (32, 20),
            (32, 21),
            (32, 22),
            (32, 23),
            (32, 24),
            (32, 25),
            (32, 26),
            (32, 27),
            (32, 28),
            (32, 29),
            (32, 30),
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
            (33, 18),
            (33, 19),
            (33, 20),
            (33, 21),
            (33, 22),
            (33, 23),
            (33, 24),
            (33, 25),
            (33, 26),
            (33, 27),
            (33, 28),
            (33, 29),
            (33, 30),
            (33, 31),
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
            (34, 18),
            (34, 19),
            (34, 20),
            (34, 21),
            (34, 22),
            (34, 23),
            (34, 24),
            (34, 25),
            (34, 26),
            (34, 27),
            (34, 28),
            (34, 29),
            (34, 30),
            (34, 31),
            (34, 32),
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
            (35, 19),
            (35, 20),
            (35, 21),
            (35, 22),
            (35, 23),
            (35, 24),
            (35, 25),
            (35, 26),
            (35, 27),
            (35, 28),
            (35, 29),
            (35, 30),
            (35, 31),
            (35, 32),
            (35, 33),
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
            (36, 19),
            (36, 20),
            (36, 21),
            (36, 22),
            (36, 23),
            (36, 24),
            (36, 25),
            (36, 26),
            (36, 27),
            (36, 28),
            (36, 29),
            (36, 30),
            (36, 31),
            (36, 32),
            (36, 33),
            (36, 34),
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
            (37, 20),
            (37, 21),
            (37, 22),
            (37, 23),
            (37, 24),
            (37, 25),
            (37, 26),
            (37, 27),
            (37, 28),
            (37, 29),
            (37, 30),
            (37, 31),
            (37, 32),
            (37, 33),
            (37, 34),
            (37, 35),
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
            (38, 21),
            (38, 22),
            (38, 23),
            (38, 24),
            (38, 25),
            (38, 26),
            (38, 27),
            (38, 28),
            (38, 29),
            (38, 30),
            (38, 31),
            (38, 32),
            (38, 33),
            (38, 34),
            (38, 35),
            (38, 36),
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
            (39, 21),
            (39, 22),
            (39, 23),
            (39, 24),
            (39, 25),
            (39, 26),
            (39, 27),
            (39, 28),
            (39, 29),
            (39, 30),
            (39, 31),
            (39, 32),
            (39, 33),
            (39, 34),
            (39, 35),
            (39, 36),
            (39, 37),
        }:
            return 19
        return 3

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_3_output, num_attn_1_3_output):
        key = (num_attn_0_3_output, num_attn_1_3_output)
        return 16

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"1", "0"}:
            return k_token == "3"
        elif q_token in {"2"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "5"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"5"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 1, 5, 16}:
            return token == "4"
        elif mlp_0_0_output in {9, 2, 3}:
            return token == "0"
        elif mlp_0_0_output in {4, 10, 13, 14, 15, 19}:
            return token == ""
        elif mlp_0_0_output in {6}:
            return token == "3"
        elif mlp_0_0_output in {12, 7}:
            return token == "<pad>"
        elif mlp_0_0_output in {8, 18}:
            return token == "1"
        elif mlp_0_0_output in {11}:
            return token == "5"
        elif mlp_0_0_output in {17}:
            return token == "2"

    attn_2_1_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 16, 12}:
            return k_position == 18
        elif q_position in {1, 5}:
            return k_position == 2
        elif q_position in {9, 2, 3, 6}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {11, 7}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {10, 14}:
            return k_position == 6
        elif q_position in {17, 18, 13, 15}:
            return k_position == 15
        elif q_position in {19}:
            return k_position == 19

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, num_mlp_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"5", "0"}:
            return k_token == "2"
        elif q_token in {"4", "1"}:
            return k_token == "5"
        elif q_token in {"2"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_0_output, attn_0_0_output):
        if attn_1_0_output in {0, 2, 11, 15}:
            return attn_0_0_output == 12
        elif attn_1_0_output in {1, 5}:
            return attn_0_0_output == 0
        elif attn_1_0_output in {3}:
            return attn_0_0_output == 3
        elif attn_1_0_output in {4}:
            return attn_0_0_output == 18
        elif attn_1_0_output in {6, 7, 8, 9, 10, 13, 14, 16, 18, 19}:
            return attn_0_0_output == 9
        elif attn_1_0_output in {17, 12}:
            return attn_0_0_output == 8

    num_attn_2_0_pattern = select(attn_0_0_outputs, attn_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_0_1_output, mlp_1_1_output):
        if mlp_0_1_output in {0, 1, 2, 3, 4, 5, 12}:
            return mlp_1_1_output == 12
        elif mlp_0_1_output in {6}:
            return mlp_1_1_output == 9
        elif mlp_0_1_output in {7}:
            return mlp_1_1_output == 6
        elif mlp_0_1_output in {8}:
            return mlp_1_1_output == 11
        elif mlp_0_1_output in {9}:
            return mlp_1_1_output == 2
        elif mlp_0_1_output in {10, 19}:
            return mlp_1_1_output == 0
        elif mlp_0_1_output in {18, 11}:
            return mlp_1_1_output == 5
        elif mlp_0_1_output in {16, 13}:
            return mlp_1_1_output == 10
        elif mlp_0_1_output in {14}:
            return mlp_1_1_output == 17
        elif mlp_0_1_output in {15}:
            return mlp_1_1_output == 8
        elif mlp_0_1_output in {17}:
            return mlp_1_1_output == 14

    num_attn_2_1_pattern = select(mlp_1_1_outputs, mlp_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "0"
        elif attn_0_3_output in {"1"}:
            return token == "1"
        elif attn_0_3_output in {"2"}:
            return token == "2"
        elif attn_0_3_output in {"3"}:
            return token == "3"
        elif attn_0_3_output in {"4", "<s>"}:
            return token == "4"
        elif attn_0_3_output in {"5"}:
            return token == "5"

    num_attn_2_2_pattern = select(tokens, attn_0_3_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_3_output, attn_0_1_output):
        if attn_1_3_output in {0, 10, 13, 14, 16, 18}:
            return attn_0_1_output == 0
        elif attn_1_3_output in {1, 4, 17}:
            return attn_0_1_output == 4
        elif attn_1_3_output in {2, 3, 12}:
            return attn_0_1_output == 3
        elif attn_1_3_output in {19, 5}:
            return attn_0_1_output == 5
        elif attn_1_3_output in {6}:
            return attn_0_1_output == 19
        elif attn_1_3_output in {8, 7}:
            return attn_0_1_output == 10
        elif attn_1_3_output in {9}:
            return attn_0_1_output == 16
        elif attn_1_3_output in {11}:
            return attn_0_1_output == 11
        elif attn_1_3_output in {15}:
            return attn_0_1_output == 17

    num_attn_2_3_pattern = select(attn_0_1_outputs, attn_1_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, token):
        key = (attn_2_3_output, token)
        return 7

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, tokens)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output, attn_2_2_output):
        key = (attn_2_1_output, attn_2_2_output)
        if key in {(1, 19), (6, 19), (10, 19), (12, 19), (13, 19), (16, 19), (18, 19)}:
            return 19
        return 12

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_3_output, num_attn_2_1_output):
        key = (num_attn_1_3_output, num_attn_2_1_output)
        return 10

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_0_output, num_attn_2_3_output):
        key = (num_attn_1_0_output, num_attn_2_3_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
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
            (7, 7),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (9, 8),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (12, 9),
            (12, 10),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (13, 10),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 7),
            (14, 8),
            (14, 9),
            (14, 10),
            (14, 11),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 10),
            (16, 11),
            (16, 12),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 9),
            (17, 10),
            (17, 11),
            (17, 12),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (18, 9),
            (18, 10),
            (18, 11),
            (18, 12),
            (18, 13),
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
            (19, 10),
            (19, 11),
            (19, 12),
            (19, 13),
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
            (20, 10),
            (20, 11),
            (20, 12),
            (20, 13),
            (20, 14),
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
            (21, 11),
            (21, 12),
            (21, 13),
            (21, 14),
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
            (22, 12),
            (22, 13),
            (22, 14),
            (22, 15),
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
            (23, 12),
            (23, 13),
            (23, 14),
            (23, 15),
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
            (24, 13),
            (24, 14),
            (24, 15),
            (24, 16),
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
            (25, 13),
            (25, 14),
            (25, 15),
            (25, 16),
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
            (26, 16),
            (26, 17),
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
            (27, 14),
            (27, 15),
            (27, 16),
            (27, 17),
            (27, 18),
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
            (28, 15),
            (28, 16),
            (28, 17),
            (28, 18),
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
            (29, 15),
            (29, 16),
            (29, 17),
            (29, 18),
            (29, 19),
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
            (30, 16),
            (30, 17),
            (30, 18),
            (30, 19),
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
            (31, 17),
            (31, 18),
            (31, 19),
            (31, 20),
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
            (32, 17),
            (32, 18),
            (32, 19),
            (32, 20),
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
            (33, 18),
            (33, 19),
            (33, 20),
            (33, 21),
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
            (34, 18),
            (34, 19),
            (34, 20),
            (34, 21),
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
            (35, 19),
            (35, 20),
            (35, 21),
            (35, 22),
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
            (36, 19),
            (36, 20),
            (36, 21),
            (36, 22),
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
            (37, 20),
            (37, 21),
            (37, 22),
            (37, 23),
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
            (38, 21),
            (38, 22),
            (38, 23),
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
            (39, 21),
            (39, 22),
            (39, 23),
            (39, 24),
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
            (40, 22),
            (40, 23),
            (40, 24),
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
            (41, 22),
            (41, 23),
            (41, 24),
            (41, 25),
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
            (42, 23),
            (42, 24),
            (42, 25),
            (42, 26),
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
            (43, 23),
            (43, 24),
            (43, 25),
            (43, 26),
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
            (44, 24),
            (44, 25),
            (44, 26),
            (44, 27),
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
            (45, 24),
            (45, 25),
            (45, 26),
            (45, 27),
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
            (46, 25),
            (46, 26),
            (46, 27),
            (46, 28),
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
            (47, 26),
            (47, 27),
            (47, 28),
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
            (48, 26),
            (48, 27),
            (48, 28),
            (48, 29),
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
            (49, 27),
            (49, 28),
            (49, 29),
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
            (50, 27),
            (50, 28),
            (50, 29),
            (50, 30),
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
            (51, 28),
            (51, 29),
            (51, 30),
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
            (52, 28),
            (52, 29),
            (52, 30),
            (52, 31),
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
            (53, 29),
            (53, 30),
            (53, 31),
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
            (54, 30),
            (54, 31),
            (54, 32),
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
            (55, 30),
            (55, 31),
            (55, 32),
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
            (56, 31),
            (56, 32),
            (56, 33),
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
            (57, 31),
            (57, 32),
            (57, 33),
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
            (58, 32),
            (58, 33),
            (58, 34),
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
            (59, 32),
            (59, 33),
            (59, 34),
            (59, 35),
        }:
            return 10
        return 9

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_3_outputs)
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
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
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


print(run(["<s>", "5", "1", "0", "0", "2", "1", "2", "4"]))
