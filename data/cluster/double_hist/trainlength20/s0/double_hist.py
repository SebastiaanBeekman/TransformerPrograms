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
        "output/length/rasp/double_hist/trainlength20/s0/double_hist_weights.csv",
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
        if position in {0, 23}:
            return token == ""
        elif position in {1}:
            return token == "5"
        elif position in {2}:
            return token == "2"
        elif position in {3}:
            return token == "1"
        elif position in {4}:
            return token == "0"
        elif position in {5, 7, 10, 11, 12, 17, 20, 21, 22, 24, 25, 26, 27, 28, 29}:
            return token == "3"
        elif position in {6, 8, 9, 13, 14, 15, 16, 18, 19}:
            return token == "4"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 5, 12, 24, 25}:
            return token == "4"
        elif position in {1, 2, 3, 4, 6, 7, 8, 20, 21, 22, 27, 29}:
            return token == ""
        elif position in {9, 10, 11, 13, 14, 16, 18, 23, 26, 28}:
            return token == "3"
        elif position in {15}:
            return token == "1"
        elif position in {17, 19}:
            return token == "0"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 2, 3, 4, 6, 7}:
            return k_position == 7
        elif q_position in {20, 5}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 2
        elif q_position in {9, 13, 14, 15}:
            return k_position == 8
        elif q_position in {10, 11, 16, 17, 19, 25}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {18}:
            return k_position == 12
        elif q_position in {21, 22, 23, 24, 26, 29}:
            return k_position == 5
        elif q_position in {27}:
            return k_position == 15
        elif q_position in {28}:
            return k_position == 16

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 11, 15}:
            return k_position == 3
        elif q_position in {8, 1, 5}:
            return k_position == 10
        elif q_position in {2}:
            return k_position == 15
        elif q_position in {3}:
            return k_position == 16
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 13
        elif q_position in {7}:
            return k_position == 20
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10, 19, 13}:
            return k_position == 7
        elif q_position in {12, 22, 24, 26, 28, 29}:
            return k_position == 4
        elif q_position in {14}:
            return k_position == 2
        elif q_position in {16, 21}:
            return k_position == 1
        elif q_position in {17, 20, 25, 23}:
            return k_position == 5
        elif q_position in {18}:
            return k_position == 27
        elif q_position in {27}:
            return k_position == 29

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 19}:
            return k_position == 7
        elif q_position in {1, 9}:
            return k_position == 15
        elif q_position in {2, 11, 12, 6}:
            return k_position == 16
        elif q_position in {10, 3, 4, 7}:
            return k_position == 14
        elif q_position in {16, 13, 5}:
            return k_position == 17
        elif q_position in {8, 26}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 24
        elif q_position in {15}:
            return k_position == 6
        elif q_position in {17, 18}:
            return k_position == 19
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {25, 29, 22}:
            return k_position == 20
        elif q_position in {23}:
            return k_position == 9
        elif q_position in {24, 27}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 18

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
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
            return k_token == ""

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 28}:
            return k_position == 15
        elif q_position in {1, 20, 5, 6}:
            return k_position == 13
        elif q_position in {9, 2, 3, 7}:
            return k_position == 11
        elif q_position in {24, 4, 14}:
            return k_position == 17
        elif q_position in {8}:
            return k_position == 27
        elif q_position in {25, 10, 26, 12}:
            return k_position == 9
        elif q_position in {17, 11, 13, 15}:
            return k_position == 26
        elif q_position in {16, 19}:
            return k_position == 6
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 14
        elif q_position in {22}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 3
        elif q_position in {27}:
            return k_position == 20
        elif q_position in {29}:
            return k_position == 28

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
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
            return k_token == ""

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "5"),
            ("3", "<s>"),
        }:
            return 9
        return 4

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, attn_0_3_output):
        key = (attn_0_1_output, attn_0_3_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "5"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "5"),
            ("1", "<s>"),
            ("3", "0"),
            ("3", "2"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "2"),
            ("4", "<s>"),
            ("5", "0"),
            ("5", "1"),
            ("5", "2"),
            ("5", "3"),
            ("5", "5"),
            ("5", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 27
        elif key in {("2", "0"), ("2", "1"), ("2", "2"), ("2", "3"), ("2", "<s>")}:
            return 10
        return 1

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_3_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 26

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 17

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_token, k_token):
        if q_token in {"2", "4", "5", "0"}:
            return k_token == "<s>"
        elif q_token in {"1", "<s>"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "4"

    attn_1_0_pattern = select_closest(tokens, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 2, 29}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 28}:
            return k_position == 5
        elif q_position in {5, 6, 7}:
            return k_position == 7
        elif q_position in {8, 9, 10, 13, 14, 18}:
            return k_position == 8
        elif q_position in {19, 26, 11, 15}:
            return k_position == 10
        elif q_position in {12, 22, 23, 24, 25}:
            return k_position == 12
        elif q_position in {16}:
            return k_position == 13
        elif q_position in {17, 27}:
            return k_position == 14
        elif q_position in {20, 21}:
            return k_position == 15

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, positions)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"2", "4", "0"}:
            return k_token == "5"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"3", "<s>"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "0"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, mlp_0_0_output):
        if token in {"4", "3", "1", "5", "0"}:
            return mlp_0_0_output == 28
        elif token in {"2", "<s>"}:
            return mlp_0_0_output == 22

    attn_1_3_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_2_output, position):
        if attn_0_2_output in {0, 1}:
            return position == 13
        elif attn_0_2_output in {2, 12, 14}:
            return position == 22
        elif attn_0_2_output in {3, 15}:
            return position == 14
        elif attn_0_2_output in {9, 4}:
            return position == 29
        elif attn_0_2_output in {17, 5, 6, 7}:
            return position == 10
        elif attn_0_2_output in {8, 25, 29, 22}:
            return position == 21
        elif attn_0_2_output in {10, 23}:
            return position == 8
        elif attn_0_2_output in {11}:
            return position == 0
        elif attn_0_2_output in {20, 13}:
            return position == 12
        elif attn_0_2_output in {16}:
            return position == 23
        elif attn_0_2_output in {18, 19, 28}:
            return position == 25
        elif attn_0_2_output in {21}:
            return position == 18
        elif attn_0_2_output in {24}:
            return position == 16
        elif attn_0_2_output in {26}:
            return position == 19
        elif attn_0_2_output in {27}:
            return position == 26

    num_attn_1_0_pattern = select(positions, attn_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_1_output, token):
        if mlp_0_1_output in {
            0,
            1,
            2,
            3,
            5,
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
        elif mlp_0_1_output in {4, 6}:
            return token == "<pad>"

    num_attn_1_1_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, attn_0_0_output):
        if attn_0_1_output in {"0"}:
            return attn_0_0_output == "5"
        elif attn_0_1_output in {"2", "1", "5"}:
            return attn_0_0_output == "0"
        elif attn_0_1_output in {"3", "4", "<s>"}:
            return attn_0_0_output == ""

    num_attn_1_2_pattern = select(attn_0_0_outputs, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_1_output, attn_0_0_output):
        if attn_0_1_output in {"0"}:
            return attn_0_0_output == "0"
        elif attn_0_1_output in {"1"}:
            return attn_0_0_output == "1"
        elif attn_0_1_output in {"2"}:
            return attn_0_0_output == "2"
        elif attn_0_1_output in {"3", "4", "<s>"}:
            return attn_0_0_output == "<s>"
        elif attn_0_1_output in {"5"}:
            return attn_0_0_output == "5"

    num_attn_1_3_pattern = select(attn_0_0_outputs, attn_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, num_mlp_0_0_output):
        key = (attn_1_3_output, num_mlp_0_0_output)
        if key in {
            (11, 2),
            (11, 4),
            (11, 10),
            (11, 11),
            (11, 15),
            (11, 16),
            (11, 17),
            (11, 19),
            (11, 24),
            (11, 29),
            (27, 4),
            (27, 10),
            (27, 11),
            (27, 15),
            (27, 19),
            (27, 29),
            (29, 0),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (29, 8),
            (29, 10),
            (29, 11),
            (29, 12),
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
            (29, 28),
            (29, 29),
        }:
            return 20
        elif key in {
            (6, 8),
            (6, 16),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 25),
            (6, 27),
            (7, 8),
            (7, 22),
            (7, 27),
            (9, 8),
            (9, 22),
            (10, 8),
            (10, 22),
            (11, 8),
            (11, 21),
            (11, 22),
            (11, 25),
            (11, 27),
            (12, 8),
            (12, 22),
            (12, 27),
            (13, 8),
            (13, 22),
            (13, 27),
            (14, 8),
            (14, 22),
            (18, 8),
            (18, 16),
            (18, 20),
            (18, 21),
            (18, 22),
            (18, 25),
            (18, 27),
            (20, 8),
            (20, 22),
            (23, 8),
            (23, 22),
            (27, 8),
            (27, 22),
            (27, 27),
        }:
            return 23
        return 13

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, num_mlp_0_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_2_output, attn_1_3_output):
        key = (attn_1_2_output, attn_1_3_output)
        if key in {
            (4, 10),
            (4, 14),
            (4, 20),
            (4, 27),
            (19, 10),
            (19, 14),
            (19, 20),
            (19, 27),
            (22, 10),
            (22, 20),
            (22, 27),
        }:
            return 0
        elif key in {(23, 12)}:
            return 8
        return 13

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_0_output, num_attn_1_2_output):
        key = (num_attn_0_0_output, num_attn_1_2_output)
        return 12

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_0_0_output):
        key = (num_attn_1_1_output, num_attn_0_0_output)
        return 7

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"2", "<s>", "3", "1", "5", "0"}:
            return k_token == ""
        elif q_token in {"4"}:
            return k_token == "4"

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_2_output, position):
        if attn_0_2_output in {0, 4}:
            return position == 4
        elif attn_0_2_output in {1, 2, 3, 6, 7}:
            return position == 5
        elif attn_0_2_output in {5, 17, 20, 21, 25, 27, 29}:
            return position == 6
        elif attn_0_2_output in {8, 9, 10}:
            return position == 8
        elif attn_0_2_output in {11, 12}:
            return position == 10
        elif attn_0_2_output in {13, 22}:
            return position == 13
        elif attn_0_2_output in {14, 23}:
            return position == 12
        elif attn_0_2_output in {15}:
            return position == 9
        elif attn_0_2_output in {16, 18, 19, 28}:
            return position == 19
        elif attn_0_2_output in {24}:
            return position == 15
        elif attn_0_2_output in {26}:
            return position == 24

    attn_2_1_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"2", "5", "0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"3", "4"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 29
        elif q_position in {1, 2, 3, 4, 5, 6, 7, 17}:
            return k_position == 5
        elif q_position in {8, 10, 20, 24, 25, 26, 27, 28, 29}:
            return k_position == 6
        elif q_position in {16, 9, 12}:
            return k_position == 8
        elif q_position in {11, 22}:
            return k_position == 10
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14, 23}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {18, 21}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 7

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(q_attn_1_1_output, k_attn_1_1_output):
        if q_attn_1_1_output in {0, 3, 20, 22, 23, 25}:
            return k_attn_1_1_output == 0
        elif q_attn_1_1_output in {1, 4, 28}:
            return k_attn_1_1_output == 4
        elif q_attn_1_1_output in {2, 29, 21}:
            return k_attn_1_1_output == 3
        elif q_attn_1_1_output in {5}:
            return k_attn_1_1_output == 5
        elif q_attn_1_1_output in {27, 10, 19, 6}:
            return k_attn_1_1_output == 21
        elif q_attn_1_1_output in {7}:
            return k_attn_1_1_output == 20
        elif q_attn_1_1_output in {8}:
            return k_attn_1_1_output == 28
        elif q_attn_1_1_output in {24, 9}:
            return k_attn_1_1_output == 29
        elif q_attn_1_1_output in {16, 17, 11}:
            return k_attn_1_1_output == 25
        elif q_attn_1_1_output in {26, 12}:
            return k_attn_1_1_output == 26
        elif q_attn_1_1_output in {18, 13}:
            return k_attn_1_1_output == 23
        elif q_attn_1_1_output in {14, 15}:
            return k_attn_1_1_output == 27

    num_attn_2_0_pattern = select(attn_1_1_outputs, attn_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, attn_0_1_output):
        if attn_1_1_output in {
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
            return attn_0_1_output == ""

    num_attn_2_1_pattern = select(attn_0_1_outputs, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_1_output, attn_1_3_output):
        if attn_0_1_output in {"2", "5", "0"}:
            return attn_1_3_output == 23
        elif attn_0_1_output in {"1"}:
            return attn_1_3_output == 26
        elif attn_0_1_output in {"3", "<s>"}:
            return attn_1_3_output == 11
        elif attn_0_1_output in {"4"}:
            return attn_1_3_output == 9

    num_attn_2_2_pattern = select(attn_1_3_outputs, attn_0_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_3_output, position):
        if attn_1_3_output in {0}:
            return position == 3
        elif attn_1_3_output in {1}:
            return position == 12
        elif attn_1_3_output in {2}:
            return position == 8
        elif attn_1_3_output in {3}:
            return position == 14
        elif attn_1_3_output in {4, 13, 15}:
            return position == 18
        elif attn_1_3_output in {18, 27, 20, 5}:
            return position == 19
        elif attn_1_3_output in {6}:
            return position == 1
        elif attn_1_3_output in {12, 22, 14, 7}:
            return position == 21
        elif attn_1_3_output in {8, 17, 21, 24, 28}:
            return position == 16
        elif attn_1_3_output in {9}:
            return position == 28
        elif attn_1_3_output in {10}:
            return position == 4
        elif attn_1_3_output in {25, 11}:
            return position == 17
        elif attn_1_3_output in {16}:
            return position == 5
        elif attn_1_3_output in {19}:
            return position == 20
        elif attn_1_3_output in {23}:
            return position == 7
        elif attn_1_3_output in {26}:
            return position == 15
        elif attn_1_3_output in {29}:
            return position == 10

    num_attn_2_3_pattern = select(positions, attn_1_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_0_1_output, attn_0_3_output):
        key = (mlp_0_1_output, attn_0_3_output)
        return 23

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, attn_0_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_2_output, attn_2_1_output):
        key = (attn_1_2_output, attn_2_1_output)
        return 2

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_1_2_output):
        key = (num_attn_2_1_output, num_attn_1_2_output)
        return 20

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_2_0_output):
        key = (num_attn_1_1_output, num_attn_2_0_output)
        return 9

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_2_0_outputs)
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


print(run(["<s>", "5", "0", "3", "3", "3", "1", "3", "5", "2", "4", "0", "0", "4"]))
