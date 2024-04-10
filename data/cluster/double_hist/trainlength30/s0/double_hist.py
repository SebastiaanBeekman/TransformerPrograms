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
        "output/length/rasp/double_hist/trainlength30/s0/double_hist_weights.csv",
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
        elif q_token in {"3", "5", "1", "4"}:
            return k_token == "4"
        elif q_token in {"2", "<s>"}:
            return k_token == "5"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "<s>"
        elif q_token in {"3", "1"}:
            return k_token == "0"
        elif q_token in {"2", "4"}:
            return k_token == "2"
        elif q_token in {"5", "<s>"}:
            return k_token == "5"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "4"
        elif q_token in {"5", "1"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"5", "1", "0"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"3", "<s>", "4"}:
            return k_token == "4"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"3", "2", "5", "0", "4"}:
            return k_token == ""
        elif q_token in {"1"}:
            return k_token == "<pad>"
        elif q_token in {"<s>"}:
            return k_token == "3"

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"3", "1", "5", "<s>", "0", "4"}:
            return position == 9
        elif token in {"2"}:
            return position == 8

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
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

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
        if q_token in {"0", "<s>"}:
            return k_token == "0"
        elif q_token in {"3", "5", "1", "2"}:
            return k_token == ""
        elif q_token in {"4"}:
            return k_token == "1"

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_1_output):
        key = (attn_0_0_output, attn_0_1_output)
        if key in {
            ("5", "0"),
            ("5", "1"),
            ("5", "2"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("5", "<s>"),
        }:
            return 30
        return 21

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, attn_0_1_output):
        key = (attn_0_3_output, attn_0_1_output)
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
            ("1", "<s>"),
            ("2", "0"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "5"),
            ("3", "<s>"),
            ("5", "0"),
            ("5", "1"),
            ("5", "2"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("5", "<s>"),
        }:
            return 4
        return 8

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_1_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        if key in {(0, 0), (0, 1), (1, 0)}:
            return 38
        elif key in {(1, 1), (2, 0), (3, 0)}:
            return 35
        elif key in {(1, 2), (3, 1), (4, 0)}:
            return 32
        elif key in {(0, 2), (2, 1)}:
            return 34
        return 2

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 13

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "3"
        elif attn_0_1_output in {"5", "1", "<s>"}:
            return token == ""
        elif attn_0_1_output in {"2"}:
            return token == "2"
        elif attn_0_1_output in {"3"}:
            return token == "0"
        elif attn_0_1_output in {"4"}:
            return token == "4"

    attn_1_0_pattern = select_closest(tokens, attn_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {32, 1, 2, 3, 4, 36}:
            return k_position == 5
        elif q_position in {5, 6}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 8
        elif q_position in {9}:
            return k_position == 9
        elif q_position in {10, 11, 13, 15}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 14
        elif q_position in {16, 17, 21, 22, 25}:
            return k_position == 13
        elif q_position in {18, 19, 20}:
            return k_position == 15
        elif q_position in {24, 26, 37, 23}:
            return k_position == 19
        elif q_position in {27}:
            return k_position == 25
        elif q_position in {28, 29}:
            return k_position == 17
        elif q_position in {30}:
            return k_position == 32
        elif q_position in {33, 31}:
            return k_position == 1
        elif q_position in {34}:
            return k_position == 37
        elif q_position in {35}:
            return k_position == 27
        elif q_position in {38}:
            return k_position == 35
        elif q_position in {39}:
            return k_position == 0

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, positions)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, num_mlp_0_0_output):
        if token in {"3", "1", "0", "4"}:
            return num_mlp_0_0_output == 2
        elif token in {"2"}:
            return num_mlp_0_0_output == 32
        elif token in {"5"}:
            return num_mlp_0_0_output == 31
        elif token in {"<s>"}:
            return num_mlp_0_0_output == 34

    attn_1_2_pattern = select_closest(num_mlp_0_0_outputs, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, num_mlp_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, mlp_0_0_output):
        if token in {"0", "4"}:
            return mlp_0_0_output == 13
        elif token in {"1", "2"}:
            return mlp_0_0_output == 21
        elif token in {"3"}:
            return mlp_0_0_output == 20
        elif token in {"5", "<s>"}:
            return mlp_0_0_output == 15

    attn_1_3_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, num_mlp_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_3_output, token):
        if attn_0_3_output in {"3", "1", "2", "5", "<s>", "0", "4"}:
            return token == ""

    num_attn_1_0_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_0_output, attn_0_1_output):
        if num_mlp_0_0_output in {0, 1, 2, 3, 33, 5, 6, 39, 10, 11, 13, 16, 20, 21, 31}:
            return attn_0_1_output == ""
        elif num_mlp_0_0_output in {32, 4, 36, 38, 15, 25}:
            return attn_0_1_output == "3"
        elif num_mlp_0_0_output in {34, 7, 18, 22, 24, 26, 27, 28, 29}:
            return attn_0_1_output == "5"
        elif num_mlp_0_0_output in {35, 8, 9, 12, 14, 17, 19, 23, 30}:
            return attn_0_1_output == "2"
        elif num_mlp_0_0_output in {37}:
            return attn_0_1_output == "0"

    num_attn_1_1_pattern = select(
        attn_0_1_outputs, num_mlp_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"3", "1", "2", "5", "<s>", "0", "4"}:
            return attn_0_2_output == ""

    num_attn_1_2_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_0_output, attn_0_1_output):
        if num_mlp_0_0_output in {
            0,
            1,
            2,
            4,
            6,
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
            25,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            39,
        }:
            return attn_0_1_output == ""
        elif num_mlp_0_0_output in {3, 5, 7, 8, 9, 24, 26}:
            return attn_0_1_output == "<pad>"
        elif num_mlp_0_0_output in {27}:
            return attn_0_1_output == "3"
        elif num_mlp_0_0_output in {38}:
            return attn_0_1_output == "<s>"

    num_attn_1_3_pattern = select(
        attn_0_1_outputs, num_mlp_0_0_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        if key in {("3", "3"), ("3", "<s>")}:
            return 10
        return 36

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_0_output, attn_1_3_output):
        key = (num_mlp_0_0_output, attn_1_3_output)
        if key in {
            (0, 28),
            (0, 35),
            (1, 3),
            (1, 5),
            (1, 7),
            (1, 13),
            (1, 14),
            (1, 17),
            (1, 19),
            (1, 27),
            (1, 28),
            (1, 35),
            (3, 28),
            (3, 35),
            (4, 5),
            (4, 17),
            (4, 28),
            (4, 35),
            (5, 5),
            (5, 13),
            (5, 17),
            (5, 28),
            (5, 35),
            (6, 3),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 13),
            (6, 14),
            (6, 17),
            (6, 19),
            (6, 23),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 35),
            (7, 28),
            (7, 35),
            (8, 28),
            (8, 35),
            (9, 28),
            (9, 35),
            (10, 3),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 13),
            (10, 14),
            (10, 17),
            (10, 19),
            (10, 23),
            (10, 27),
            (10, 28),
            (10, 29),
            (10, 35),
            (12, 28),
            (12, 35),
            (13, 5),
            (13, 28),
            (13, 35),
            (14, 28),
            (14, 35),
            (15, 28),
            (15, 35),
            (16, 28),
            (16, 35),
            (17, 28),
            (17, 35),
            (18, 28),
            (18, 35),
            (19, 28),
            (19, 35),
            (20, 3),
            (20, 5),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 13),
            (20, 14),
            (20, 17),
            (20, 19),
            (20, 23),
            (20, 24),
            (20, 26),
            (20, 27),
            (20, 28),
            (20, 29),
            (20, 33),
            (20, 35),
            (21, 28),
            (21, 35),
            (22, 28),
            (22, 35),
            (23, 28),
            (23, 35),
            (24, 35),
            (25, 28),
            (25, 35),
            (26, 28),
            (26, 35),
            (27, 28),
            (27, 35),
            (28, 28),
            (28, 35),
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
            (29, 28),
            (29, 29),
            (29, 30),
            (29, 31),
            (29, 32),
            (29, 33),
            (29, 34),
            (29, 35),
            (29, 36),
            (29, 38),
            (29, 39),
            (30, 28),
            (30, 35),
            (33, 5),
            (33, 28),
            (33, 35),
            (35, 28),
            (35, 35),
            (36, 28),
            (36, 35),
            (39, 28),
            (39, 35),
        }:
            return 23
        elif key in {
            (1, 4),
            (1, 6),
            (1, 8),
            (1, 15),
            (1, 18),
            (1, 21),
            (1, 23),
            (1, 24),
            (1, 26),
            (1, 29),
            (1, 33),
            (1, 39),
            (3, 5),
            (3, 7),
            (3, 13),
            (3, 14),
            (3, 17),
            (3, 19),
            (3, 27),
            (4, 3),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 13),
            (4, 14),
            (4, 19),
            (4, 21),
            (4, 23),
            (4, 24),
            (4, 26),
            (4, 27),
            (4, 29),
            (4, 33),
            (5, 3),
            (5, 4),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 14),
            (5, 19),
            (5, 21),
            (5, 23),
            (5, 24),
            (5, 26),
            (5, 27),
            (5, 29),
            (5, 33),
            (5, 39),
            (6, 4),
            (6, 12),
            (6, 15),
            (6, 18),
            (6, 20),
            (6, 21),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 36),
            (6, 38),
            (6, 39),
            (10, 4),
            (10, 12),
            (10, 15),
            (10, 18),
            (10, 20),
            (10, 21),
            (10, 24),
            (10, 25),
            (10, 26),
            (10, 32),
            (10, 33),
            (10, 34),
            (10, 36),
            (10, 38),
            (10, 39),
            (11, 28),
            (11, 35),
            (13, 3),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 13),
            (13, 14),
            (13, 17),
            (13, 19),
            (13, 27),
            (13, 29),
            (14, 5),
            (14, 13),
            (14, 17),
            (16, 5),
            (16, 7),
            (16, 13),
            (16, 14),
            (16, 17),
            (16, 19),
            (16, 27),
            (17, 5),
            (17, 13),
            (17, 17),
            (20, 4),
            (20, 9),
            (20, 12),
            (20, 15),
            (20, 18),
            (20, 20),
            (20, 21),
            (20, 25),
            (20, 30),
            (20, 32),
            (20, 34),
            (20, 36),
            (20, 38),
            (20, 39),
            (23, 5),
            (23, 13),
            (23, 17),
            (24, 28),
            (26, 3),
            (26, 5),
            (26, 7),
            (26, 8),
            (26, 13),
            (26, 14),
            (26, 17),
            (26, 19),
            (26, 27),
            (29, 37),
            (30, 5),
            (30, 13),
            (30, 17),
            (33, 3),
            (33, 6),
            (33, 7),
            (33, 8),
            (33, 13),
            (33, 14),
            (33, 17),
            (33, 19),
            (33, 23),
            (33, 27),
            (33, 29),
            (33, 33),
        }:
            return 31
        elif key in {
            (0, 11),
            (1, 11),
            (19, 11),
            (25, 11),
            (28, 11),
            (32, 0),
            (32, 1),
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
            (32, 16),
            (32, 17),
            (32, 18),
            (32, 20),
            (32, 21),
            (32, 23),
            (32, 24),
            (32, 25),
            (32, 26),
            (32, 28),
            (32, 30),
            (32, 32),
            (32, 33),
            (32, 36),
            (32, 37),
            (32, 38),
            (32, 39),
        }:
            return 9
        elif key in {(6, 11), (10, 11), (20, 11)}:
            return 35
        return 38

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_0_output, num_attn_1_0_output):
        key = (num_attn_0_0_output, num_attn_1_0_output)
        return 26

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_3_output):
        key = num_attn_1_3_output
        return 14

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_1_3_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "4"
        elif q_token in {"5", "1", "4"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, num_mlp_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 14
        elif q_position in {1, 2, 3, 4, 6}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {8, 35, 38, 7}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 1
        elif q_position in {10, 27}:
            return k_position == 8
        elif q_position in {32, 11, 13, 18, 20, 21}:
            return k_position == 10
        elif q_position in {12, 14, 16, 17, 24, 29}:
            return k_position == 6
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {19, 36}:
            return k_position == 11
        elif q_position in {34, 22}:
            return k_position == 20
        elif q_position in {28, 23}:
            return k_position == 15
        elif q_position in {25}:
            return k_position == 9
        elif q_position in {33, 26}:
            return k_position == 19
        elif q_position in {30}:
            return k_position == 4
        elif q_position in {31}:
            return k_position == 18
        elif q_position in {37}:
            return k_position == 24
        elif q_position in {39}:
            return k_position == 25

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 34, 39, 10, 14, 19, 20, 26}:
            return k_num_mlp_0_0_output == 8
        elif q_num_mlp_0_0_output in {
            1,
            33,
            3,
            4,
            5,
            6,
            36,
            8,
            9,
            12,
            17,
            21,
            22,
            24,
            25,
            27,
            28,
            30,
        }:
            return k_num_mlp_0_0_output == 2
        elif q_num_mlp_0_0_output in {2, 31, 13, 23}:
            return k_num_mlp_0_0_output == 31
        elif q_num_mlp_0_0_output in {18, 7}:
            return k_num_mlp_0_0_output == 23
        elif q_num_mlp_0_0_output in {11}:
            return k_num_mlp_0_0_output == 36
        elif q_num_mlp_0_0_output in {16, 15}:
            return k_num_mlp_0_0_output == 22
        elif q_num_mlp_0_0_output in {32, 35, 29}:
            return k_num_mlp_0_0_output == 34
        elif q_num_mlp_0_0_output in {37}:
            return k_num_mlp_0_0_output == 21
        elif q_num_mlp_0_0_output in {38}:
            return k_num_mlp_0_0_output == 4

    attn_2_2_pattern = select_closest(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, predicate_2_2
    )
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 1, 3, 6}:
            return position == 14
        elif num_mlp_0_0_output in {2}:
            return position == 19
        elif num_mlp_0_0_output in {4, 5, 9, 24, 30}:
            return position == 11
        elif num_mlp_0_0_output in {7}:
            return position == 36
        elif num_mlp_0_0_output in {8, 12, 21, 14}:
            return position == 10
        elif num_mlp_0_0_output in {10}:
            return position == 16
        elif num_mlp_0_0_output in {16, 11}:
            return position == 13
        elif num_mlp_0_0_output in {13}:
            return position == 29
        elif num_mlp_0_0_output in {25, 15}:
            return position == 35
        elif num_mlp_0_0_output in {17, 29, 22}:
            return position == 9
        elif num_mlp_0_0_output in {18}:
            return position == 15
        elif num_mlp_0_0_output in {19}:
            return position == 8
        elif num_mlp_0_0_output in {20}:
            return position == 21
        elif num_mlp_0_0_output in {34, 35, 28, 23}:
            return position == 7
        elif num_mlp_0_0_output in {32, 26, 27}:
            return position == 12
        elif num_mlp_0_0_output in {31}:
            return position == 17
        elif num_mlp_0_0_output in {33}:
            return position == 20
        elif num_mlp_0_0_output in {36, 39}:
            return position == 18
        elif num_mlp_0_0_output in {37}:
            return position == 6
        elif num_mlp_0_0_output in {38}:
            return position == 5

    attn_2_3_pattern = select_closest(positions, num_mlp_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 36, 39, 7, 25}:
            return k_num_mlp_0_0_output == 11
        elif q_num_mlp_0_0_output in {1}:
            return k_num_mlp_0_0_output == 10
        elif q_num_mlp_0_0_output in {2}:
            return k_num_mlp_0_0_output == 22
        elif q_num_mlp_0_0_output in {3}:
            return k_num_mlp_0_0_output == 38
        elif q_num_mlp_0_0_output in {4}:
            return k_num_mlp_0_0_output == 36
        elif q_num_mlp_0_0_output in {10, 5}:
            return k_num_mlp_0_0_output == 25
        elif q_num_mlp_0_0_output in {6, 8, 12, 14, 15, 17, 18, 24, 26, 27, 29}:
            return k_num_mlp_0_0_output == 34
        elif q_num_mlp_0_0_output in {34, 9, 19, 21, 22, 30}:
            return k_num_mlp_0_0_output == 29
        elif q_num_mlp_0_0_output in {32, 11}:
            return k_num_mlp_0_0_output == 32
        elif q_num_mlp_0_0_output in {13}:
            return k_num_mlp_0_0_output == 16
        elif q_num_mlp_0_0_output in {16}:
            return k_num_mlp_0_0_output == 39
        elif q_num_mlp_0_0_output in {20, 31}:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {28, 38, 23}:
            return k_num_mlp_0_0_output == 35
        elif q_num_mlp_0_0_output in {33}:
            return k_num_mlp_0_0_output == 0
        elif q_num_mlp_0_0_output in {35, 37}:
            return k_num_mlp_0_0_output == 28

    num_attn_2_0_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, token):
        if attn_1_1_output in {0}:
            return token == "2"
        elif attn_1_1_output in {
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
            27,
            28,
            29,
            30,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif attn_1_1_output in {26, 12, 31}:
            return token == "<pad>"

    num_attn_2_1_pattern = select(tokens, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_3_output, mlp_0_1_output):
        if attn_0_3_output in {"0"}:
            return mlp_0_1_output == 24
        elif attn_0_3_output in {"1", "2"}:
            return mlp_0_1_output == 19
        elif attn_0_3_output in {"3"}:
            return mlp_0_1_output == 11
        elif attn_0_3_output in {"4"}:
            return mlp_0_1_output == 37
        elif attn_0_3_output in {"5"}:
            return mlp_0_1_output == 22
        elif attn_0_3_output in {"<s>"}:
            return mlp_0_1_output == 9

    num_attn_2_2_pattern = select(mlp_0_1_outputs, attn_0_3_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {
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
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""

    num_attn_2_3_pattern = select(tokens, num_mlp_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_2_output, mlp_0_0_output):
        key = (attn_1_2_output, mlp_0_0_output)
        if key in {
            (4, 33),
            (4, 35),
            (7, 33),
            (10, 33),
            (10, 35),
            (16, 33),
            (21, 25),
            (21, 33),
            (21, 35),
            (22, 33),
            (30, 25),
            (30, 33),
            (30, 35),
            (37, 33),
            (37, 35),
        }:
            return 30
        elif key in {(8, 16), (20, 16)}:
            return 8
        return 37

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, mlp_0_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_0_output, attn_0_3_output):
        key = (attn_0_0_output, attn_0_3_output)
        return 12

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_1_2_output):
        key = (num_attn_2_1_output, num_attn_1_2_output)
        return 24

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_1_2_output):
        key = (num_attn_1_1_output, num_attn_1_2_output)
        return 20

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_2_outputs)
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
