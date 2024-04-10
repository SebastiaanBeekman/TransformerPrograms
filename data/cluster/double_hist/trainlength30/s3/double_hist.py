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
        "output/length/rasp/double_hist/trainlength30/s3/double_hist_weights.csv",
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
        if q_token in {"<s>", "0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "3"
        elif q_token in {"5", "3", "4", "2"}:
            return k_token == "1"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "1"
        elif q_token in {"<s>", "1", "3", "5"}:
            return k_token == "0"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "4"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"3", "1", "0", "4", "5", "2"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "1"
        elif q_token in {"<s>", "1"}:
            return k_token == "5"
        elif q_token in {"2"}:
            return k_token == "4"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"4", "5"}:
            return k_token == "3"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 2, 5, 11, 31}:
            return k_position == 28
        elif q_position in {1}:
            return k_position == 23
        elif q_position in {8, 3, 4, 30}:
            return k_position == 17
        elif q_position in {17, 6}:
            return k_position == 19
        elif q_position in {7}:
            return k_position == 29
        elif q_position in {9, 18, 35, 29}:
            return k_position == 25
        elif q_position in {10}:
            return k_position == 21
        elif q_position in {12, 36}:
            return k_position == 14
        elif q_position in {26, 27, 34, 13}:
            return k_position == 30
        elif q_position in {14}:
            return k_position == 35
        elif q_position in {15}:
            return k_position == 22
        elif q_position in {16, 19, 21}:
            return k_position == 31
        elif q_position in {20}:
            return k_position == 33
        elif q_position in {32, 22}:
            return k_position == 26
        elif q_position in {23}:
            return k_position == 39
        elif q_position in {24, 28, 39}:
            return k_position == 34
        elif q_position in {25, 38}:
            return k_position == 37
        elif q_position in {33}:
            return k_position == 12
        elif q_position in {37}:
            return k_position == 24

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"1", "0", "4", "2", "<s>", "5"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "<s>"

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 35, 29, 22}:
            return k_position == 32
        elif q_position in {1, 2, 14}:
            return k_position == 22
        elif q_position in {16, 3, 12, 6}:
            return k_position == 25
        elif q_position in {4, 21}:
            return k_position == 26
        elif q_position in {13, 11, 5}:
            return k_position == 21
        elif q_position in {18, 7}:
            return k_position == 20
        elif q_position in {8}:
            return k_position == 23
        elif q_position in {39, 9, 10, 17, 26}:
            return k_position == 28
        elif q_position in {24, 15}:
            return k_position == 37
        elif q_position in {19, 20}:
            return k_position == 27
        elif q_position in {32, 33, 31, 23}:
            return k_position == 29
        elif q_position in {25}:
            return k_position == 33
        elif q_position in {34, 27, 30}:
            return k_position == 31
        elif q_position in {28, 36}:
            return k_position == 30
        elif q_position in {37}:
            return k_position == 39
        elif q_position in {38}:
            return k_position == 38

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
    def mlp_0_0(attn_0_2_output, attn_0_0_output):
        key = (attn_0_2_output, attn_0_0_output)
        if key in {("0", "0"), ("0", "<s>")}:
            return 21
        return 6

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_0_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output, attn_0_2_output):
        key = (attn_0_0_output, attn_0_2_output)
        if key in {
            ("0", "5"),
            ("0", "<s>"),
            ("5", "3"),
            ("5", "5"),
            ("5", "<s>"),
            ("<s>", "3"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 21
        elif key in {("0", "3")}:
            return 2
        return 1

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_2_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_0_output):
        key = (num_attn_0_2_output, num_attn_0_0_output)
        return 16

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 23

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_token, k_token):
        if q_token in {"<s>", "0", "2"}:
            return k_token == "5"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "<s>"
        elif q_token in {"5"}:
            return k_token == "2"

    attn_1_0_pattern = select_closest(tokens, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, mlp_0_1_output):
        if token in {"0"}:
            return mlp_0_1_output == 16
        elif token in {"1", "5"}:
            return mlp_0_1_output == 14
        elif token in {"2"}:
            return mlp_0_1_output == 12
        elif token in {"3"}:
            return mlp_0_1_output == 17
        elif token in {"4"}:
            return mlp_0_1_output == 18
        elif token in {"<s>"}:
            return mlp_0_1_output == 0

    attn_1_1_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 2, 3, 4, 5, 6}:
            return k_position == 6
        elif q_position in {7, 12, 13, 14, 17, 18, 22}:
            return k_position == 9
        elif q_position in {8, 9, 10, 11}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 11
        elif q_position in {16}:
            return k_position == 10
        elif q_position in {19, 28, 23}:
            return k_position == 17
        elif q_position in {26, 20}:
            return k_position == 13
        elif q_position in {24, 25, 34, 21}:
            return k_position == 14
        elif q_position in {27}:
            return k_position == 16
        elif q_position in {29}:
            return k_position == 19
        elif q_position in {37, 30}:
            return k_position == 24
        elif q_position in {36, 39, 31}:
            return k_position == 5
        elif q_position in {32, 35}:
            return k_position == 18
        elif q_position in {33}:
            return k_position == 33
        elif q_position in {38}:
            return k_position == 27

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, positions)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_token, k_token):
        if q_token in {"0", "4", "2"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "<s>"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_1_3_pattern = select_closest(tokens, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_1_output, attn_0_0_output):
        if attn_0_1_output in {"3", "1", "0", "4", "5", "2"}:
            return attn_0_0_output == ""
        elif attn_0_1_output in {"<s>"}:
            return attn_0_0_output == "4"

    num_attn_1_0_pattern = select(attn_0_0_outputs, attn_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"0"}:
            return mlp_0_0_output == 10
        elif attn_0_1_output in {"1"}:
            return mlp_0_0_output == 33
        elif attn_0_1_output in {"2"}:
            return mlp_0_0_output == 16
        elif attn_0_1_output in {"3"}:
            return mlp_0_0_output == 11
        elif attn_0_1_output in {"4"}:
            return mlp_0_0_output == 26
        elif attn_0_1_output in {"5"}:
            return mlp_0_0_output == 27
        elif attn_0_1_output in {"<s>"}:
            return mlp_0_0_output == 37

    num_attn_1_1_pattern = select(mlp_0_0_outputs, attn_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, token):
        if attn_0_1_output in {"3", "1", "0", "4", "5", "<s>", "2"}:
            return token == ""

    num_attn_1_2_pattern = select(tokens, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_1_output, attn_0_2_output):
        if attn_0_1_output in {"<s>", "0", "2"}:
            return attn_0_2_output == ""
        elif attn_0_1_output in {"1"}:
            return attn_0_2_output == "1"
        elif attn_0_1_output in {"3"}:
            return attn_0_2_output == "3"
        elif attn_0_1_output in {"4"}:
            return attn_0_2_output == "4"
        elif attn_0_1_output in {"5"}:
            return attn_0_2_output == "5"

    num_attn_1_3_pattern = select(attn_0_2_outputs, attn_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("1", "<s>"),
            ("2", "1"),
            ("3", "0"),
            ("3", "1"),
            ("3", "4"),
            ("3", "5"),
            ("3", "<s>"),
            ("4", "1"),
            ("5", "1"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 32
        return 23

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_0_output, num_mlp_0_1_output):
        key = (attn_1_0_output, num_mlp_0_1_output)
        if key in {
            (8, 4),
            (8, 32),
            (10, 4),
            (10, 32),
            (19, 4),
            (19, 12),
            (19, 18),
            (19, 31),
            (19, 32),
            (19, 39),
            (23, 4),
            (23, 32),
            (32, 4),
            (32, 32),
            (33, 4),
            (33, 32),
            (35, 4),
            (35, 32),
            (36, 4),
            (36, 32),
        }:
            return 9
        elif key in {(0, 8), (23, 8)}:
            return 17
        return 11

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, num_mlp_0_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_1_3_output):
        key = (num_attn_0_1_output, num_attn_1_3_output)
        return 12

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_1_1_output):
        key = (num_attn_1_0_output, num_attn_1_1_output)
        if key in {
            (12, 0),
            (13, 0),
            (14, 0),
            (15, 0),
            (16, 0),
            (16, 1),
            (17, 0),
            (17, 1),
            (18, 0),
            (18, 1),
            (19, 0),
            (19, 1),
            (19, 2),
            (20, 0),
            (20, 1),
            (20, 2),
            (21, 0),
            (21, 1),
            (21, 2),
            (22, 0),
            (22, 1),
            (22, 2),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
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
            (60, 0),
            (60, 1),
            (60, 2),
            (60, 3),
            (60, 4),
            (60, 5),
            (60, 6),
            (60, 7),
            (60, 8),
            (60, 9),
            (60, 10),
            (60, 11),
            (60, 12),
            (60, 13),
            (61, 0),
            (61, 1),
            (61, 2),
            (61, 3),
            (61, 4),
            (61, 5),
            (61, 6),
            (61, 7),
            (61, 8),
            (61, 9),
            (61, 10),
            (61, 11),
            (61, 12),
            (61, 13),
            (62, 0),
            (62, 1),
            (62, 2),
            (62, 3),
            (62, 4),
            (62, 5),
            (62, 6),
            (62, 7),
            (62, 8),
            (62, 9),
            (62, 10),
            (62, 11),
            (62, 12),
            (62, 13),
            (63, 0),
            (63, 1),
            (63, 2),
            (63, 3),
            (63, 4),
            (63, 5),
            (63, 6),
            (63, 7),
            (63, 8),
            (63, 9),
            (63, 10),
            (63, 11),
            (63, 12),
            (63, 13),
            (64, 0),
            (64, 1),
            (64, 2),
            (64, 3),
            (64, 4),
            (64, 5),
            (64, 6),
            (64, 7),
            (64, 8),
            (64, 9),
            (64, 10),
            (64, 11),
            (64, 12),
            (64, 13),
            (64, 14),
            (65, 0),
            (65, 1),
            (65, 2),
            (65, 3),
            (65, 4),
            (65, 5),
            (65, 6),
            (65, 7),
            (65, 8),
            (65, 9),
            (65, 10),
            (65, 11),
            (65, 12),
            (65, 13),
            (65, 14),
            (66, 0),
            (66, 1),
            (66, 2),
            (66, 3),
            (66, 4),
            (66, 5),
            (66, 6),
            (66, 7),
            (66, 8),
            (66, 9),
            (66, 10),
            (66, 11),
            (66, 12),
            (66, 13),
            (66, 14),
            (67, 0),
            (67, 1),
            (67, 2),
            (67, 3),
            (67, 4),
            (67, 5),
            (67, 6),
            (67, 7),
            (67, 8),
            (67, 9),
            (67, 10),
            (67, 11),
            (67, 12),
            (67, 13),
            (67, 14),
            (68, 0),
            (68, 1),
            (68, 2),
            (68, 3),
            (68, 4),
            (68, 5),
            (68, 6),
            (68, 7),
            (68, 8),
            (68, 9),
            (68, 10),
            (68, 11),
            (68, 12),
            (68, 13),
            (68, 14),
            (68, 15),
            (69, 0),
            (69, 1),
            (69, 2),
            (69, 3),
            (69, 4),
            (69, 5),
            (69, 6),
            (69, 7),
            (69, 8),
            (69, 9),
            (69, 10),
            (69, 11),
            (69, 12),
            (69, 13),
            (69, 14),
            (69, 15),
            (70, 0),
            (70, 1),
            (70, 2),
            (70, 3),
            (70, 4),
            (70, 5),
            (70, 6),
            (70, 7),
            (70, 8),
            (70, 9),
            (70, 10),
            (70, 11),
            (70, 12),
            (70, 13),
            (70, 14),
            (70, 15),
            (71, 0),
            (71, 1),
            (71, 2),
            (71, 3),
            (71, 4),
            (71, 5),
            (71, 6),
            (71, 7),
            (71, 8),
            (71, 9),
            (71, 10),
            (71, 11),
            (71, 12),
            (71, 13),
            (71, 14),
            (71, 15),
            (72, 0),
            (72, 1),
            (72, 2),
            (72, 3),
            (72, 4),
            (72, 5),
            (72, 6),
            (72, 7),
            (72, 8),
            (72, 9),
            (72, 10),
            (72, 11),
            (72, 12),
            (72, 13),
            (72, 14),
            (72, 15),
            (72, 16),
            (73, 0),
            (73, 1),
            (73, 2),
            (73, 3),
            (73, 4),
            (73, 5),
            (73, 6),
            (73, 7),
            (73, 8),
            (73, 9),
            (73, 10),
            (73, 11),
            (73, 12),
            (73, 13),
            (73, 14),
            (73, 15),
            (73, 16),
            (74, 0),
            (74, 1),
            (74, 2),
            (74, 3),
            (74, 4),
            (74, 5),
            (74, 6),
            (74, 7),
            (74, 8),
            (74, 9),
            (74, 10),
            (74, 11),
            (74, 12),
            (74, 13),
            (74, 14),
            (74, 15),
            (74, 16),
            (75, 0),
            (75, 1),
            (75, 2),
            (75, 3),
            (75, 4),
            (75, 5),
            (75, 6),
            (75, 7),
            (75, 8),
            (75, 9),
            (75, 10),
            (75, 11),
            (75, 12),
            (75, 13),
            (75, 14),
            (75, 15),
            (75, 16),
            (75, 17),
            (76, 0),
            (76, 1),
            (76, 2),
            (76, 3),
            (76, 4),
            (76, 5),
            (76, 6),
            (76, 7),
            (76, 8),
            (76, 9),
            (76, 10),
            (76, 11),
            (76, 12),
            (76, 13),
            (76, 14),
            (76, 15),
            (76, 16),
            (76, 17),
            (77, 0),
            (77, 1),
            (77, 2),
            (77, 3),
            (77, 4),
            (77, 5),
            (77, 6),
            (77, 7),
            (77, 8),
            (77, 9),
            (77, 10),
            (77, 11),
            (77, 12),
            (77, 13),
            (77, 14),
            (77, 15),
            (77, 16),
            (77, 17),
            (78, 0),
            (78, 1),
            (78, 2),
            (78, 3),
            (78, 4),
            (78, 5),
            (78, 6),
            (78, 7),
            (78, 8),
            (78, 9),
            (78, 10),
            (78, 11),
            (78, 12),
            (78, 13),
            (78, 14),
            (78, 15),
            (78, 16),
            (78, 17),
            (79, 0),
            (79, 1),
            (79, 2),
            (79, 3),
            (79, 4),
            (79, 5),
            (79, 6),
            (79, 7),
            (79, 8),
            (79, 9),
            (79, 10),
            (79, 11),
            (79, 12),
            (79, 13),
            (79, 14),
            (79, 15),
            (79, 16),
            (79, 17),
            (79, 18),
        }:
            return 6
        return 16

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_2_output, position):
        if attn_0_2_output in {"0"}:
            return position == 25
        elif attn_0_2_output in {"1"}:
            return position == 9
        elif attn_0_2_output in {"2"}:
            return position == 5
        elif attn_0_2_output in {"3"}:
            return position == 19
        elif attn_0_2_output in {"4"}:
            return position == 8
        elif attn_0_2_output in {"5"}:
            return position == 10
        elif attn_0_2_output in {"<s>"}:
            return position == 12

    attn_2_0_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_2_output, position):
        if attn_1_2_output in {0, 29, 3, 5}:
            return position == 6
        elif attn_1_2_output in {1, 18}:
            return position == 9
        elif attn_1_2_output in {2, 14}:
            return position == 20
        elif attn_1_2_output in {4, 37}:
            return position == 5
        elif attn_1_2_output in {6}:
            return position == 7
        elif attn_1_2_output in {31, 7}:
            return position == 8
        elif attn_1_2_output in {8, 36, 30}:
            return position == 11
        elif attn_1_2_output in {9}:
            return position == 10
        elif attn_1_2_output in {10, 20}:
            return position == 14
        elif attn_1_2_output in {25, 11}:
            return position == 13
        elif attn_1_2_output in {19, 12, 22, 15}:
            return position == 24
        elif attn_1_2_output in {13}:
            return position == 18
        elif attn_1_2_output in {16}:
            return position == 16
        elif attn_1_2_output in {17}:
            return position == 22
        elif attn_1_2_output in {21}:
            return position == 17
        elif attn_1_2_output in {32, 39, 23}:
            return position == 31
        elif attn_1_2_output in {24}:
            return position == 27
        elif attn_1_2_output in {26}:
            return position == 23
        elif attn_1_2_output in {27}:
            return position == 29
        elif attn_1_2_output in {28}:
            return position == 25
        elif attn_1_2_output in {33}:
            return position == 19
        elif attn_1_2_output in {34}:
            return position == 32
        elif attn_1_2_output in {35}:
            return position == 21
        elif attn_1_2_output in {38}:
            return position == 26

    attn_2_1_pattern = select_closest(positions, attn_1_2_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_1_2_output, position):
        if attn_1_2_output in {0, 33, 27}:
            return position == 23
        elif attn_1_2_output in {1}:
            return position == 10
        elif attn_1_2_output in {2}:
            return position == 12
        elif attn_1_2_output in {8, 34, 3, 36}:
            return position == 9
        elif attn_1_2_output in {4}:
            return position == 7
        elif attn_1_2_output in {5}:
            return position == 14
        elif attn_1_2_output in {13, 6, 15}:
            return position == 16
        elif attn_1_2_output in {38, 7}:
            return position == 8
        elif attn_1_2_output in {9, 19, 20}:
            return position == 13
        elif attn_1_2_output in {10}:
            return position == 18
        elif attn_1_2_output in {11, 12}:
            return position == 6
        elif attn_1_2_output in {35, 37, 14}:
            return position == 22
        elif attn_1_2_output in {16}:
            return position == 25
        elif attn_1_2_output in {17, 30}:
            return position == 26
        elif attn_1_2_output in {25, 18}:
            return position == 20
        elif attn_1_2_output in {21, 23}:
            return position == 27
        elif attn_1_2_output in {22}:
            return position == 15
        elif attn_1_2_output in {24}:
            return position == 1
        elif attn_1_2_output in {26}:
            return position == 28
        elif attn_1_2_output in {28}:
            return position == 5
        elif attn_1_2_output in {29, 39}:
            return position == 24
        elif attn_1_2_output in {31}:
            return position == 0
        elif attn_1_2_output in {32}:
            return position == 32

    attn_2_2_pattern = select_closest(positions, attn_1_2_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_0_output, token):
        if attn_0_0_output in {"<s>", "0"}:
            return token == ""
        elif attn_0_0_output in {"1"}:
            return token == "0"
        elif attn_0_0_output in {"2"}:
            return token == "4"
        elif attn_0_0_output in {"3"}:
            return token == "1"
        elif attn_0_0_output in {"4"}:
            return token == "3"
        elif attn_0_0_output in {"5"}:
            return token == "5"

    attn_2_3_pattern = select_closest(tokens, attn_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_0_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_2_output, token):
        if attn_0_2_output in {"<s>", "5", "0", "2"}:
            return token == ""
        elif attn_0_2_output in {"1", "4"}:
            return token == "3"
        elif attn_0_2_output in {"3"}:
            return token == "5"

    num_attn_2_0_pattern = select(tokens, attn_0_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_0_1_output, attn_0_0_output):
        if mlp_0_1_output in {
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
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return attn_0_0_output == ""
        elif mlp_0_1_output in {20}:
            return attn_0_0_output == "<pad>"
        elif mlp_0_1_output in {33}:
            return attn_0_0_output == "3"

    num_attn_2_1_pattern = select(attn_0_0_outputs, mlp_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"<s>", "0"}:
            return mlp_0_0_output == 21
        elif attn_0_1_output in {"1", "5"}:
            return mlp_0_0_output == 8
        elif attn_0_1_output in {"2"}:
            return mlp_0_0_output == 11
        elif attn_0_1_output in {"3"}:
            return mlp_0_0_output == 15
        elif attn_0_1_output in {"4"}:
            return mlp_0_0_output == 22

    num_attn_2_2_pattern = select(mlp_0_0_outputs, attn_0_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_2_output, token):
        if attn_0_2_output in {"1", "0", "5"}:
            return token == ""
        elif attn_0_2_output in {"2"}:
            return token == "2"
        elif attn_0_2_output in {"3"}:
            return token == "3"
        elif attn_0_2_output in {"4"}:
            return token == "4"
        elif attn_0_2_output in {"<s>"}:
            return token == "<pad>"

    num_attn_2_3_pattern = select(tokens, attn_0_2_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_1_0_output, attn_2_2_output):
        key = (num_mlp_1_0_output, attn_2_2_output)
        if key in {
            (9, 19),
            (22, 2),
            (22, 3),
            (22, 7),
            (22, 9),
            (22, 10),
            (22, 15),
            (22, 17),
            (22, 18),
            (22, 19),
            (22, 20),
            (22, 23),
            (22, 24),
            (22, 25),
            (22, 29),
            (22, 30),
            (22, 31),
            (22, 34),
            (22, 35),
            (22, 36),
            (22, 37),
            (22, 38),
            (22, 39),
            (27, 19),
            (28, 18),
            (28, 19),
            (28, 23),
            (28, 25),
            (28, 35),
            (29, 19),
            (33, 18),
            (33, 19),
            (33, 23),
            (36, 18),
            (36, 19),
        }:
            return 3
        elif key in {(23, 24)}:
            return 33
        return 11

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_1_0_outputs, attn_2_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_1_output, num_mlp_0_0_output):
        key = (attn_1_1_output, num_mlp_0_0_output)
        if key in {
            (0, 9),
            (0, 13),
            (0, 15),
            (0, 28),
            (2, 13),
            (2, 28),
            (3, 13),
            (3, 15),
            (3, 28),
            (4, 13),
            (4, 15),
            (4, 28),
            (5, 13),
            (5, 15),
            (5, 28),
            (6, 13),
            (6, 15),
            (6, 28),
            (7, 13),
            (7, 15),
            (7, 28),
            (8, 13),
            (8, 15),
            (8, 28),
            (9, 4),
            (9, 9),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 28),
            (9, 31),
            (9, 33),
            (9, 37),
            (9, 39),
            (10, 9),
            (10, 13),
            (10, 15),
            (10, 28),
            (11, 13),
            (12, 3),
            (12, 13),
            (12, 15),
            (12, 28),
            (12, 33),
            (13, 13),
            (13, 15),
            (13, 28),
            (15, 13),
            (18, 4),
            (18, 9),
            (18, 13),
            (18, 14),
            (18, 15),
            (18, 28),
            (18, 31),
            (18, 33),
            (18, 37),
            (18, 39),
            (19, 3),
            (19, 13),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 9),
            (20, 10),
            (20, 13),
            (20, 14),
            (20, 15),
            (20, 17),
            (20, 18),
            (20, 19),
            (20, 20),
            (20, 22),
            (20, 25),
            (20, 26),
            (20, 28),
            (20, 29),
            (20, 30),
            (20, 31),
            (20, 32),
            (20, 33),
            (20, 34),
            (20, 35),
            (20, 37),
            (20, 38),
            (20, 39),
            (21, 13),
            (21, 15),
            (21, 28),
            (22, 13),
            (22, 15),
            (22, 28),
            (23, 3),
            (23, 13),
            (24, 4),
            (24, 9),
            (24, 13),
            (24, 15),
            (24, 28),
            (24, 33),
            (24, 37),
            (25, 9),
            (25, 13),
            (25, 15),
            (25, 28),
            (26, 3),
            (26, 13),
            (26, 28),
            (26, 33),
            (27, 4),
            (27, 9),
            (27, 13),
            (27, 14),
            (27, 15),
            (27, 18),
            (27, 28),
            (27, 31),
            (27, 33),
            (27, 37),
            (27, 39),
            (28, 3),
            (28, 13),
            (29, 13),
            (29, 15),
            (29, 28),
            (30, 13),
            (30, 15),
            (30, 28),
            (31, 13),
            (32, 3),
            (32, 13),
            (32, 28),
            (33, 13),
            (33, 15),
            (33, 28),
            (34, 3),
            (34, 4),
            (34, 9),
            (34, 13),
            (34, 14),
            (34, 15),
            (34, 18),
            (34, 28),
            (34, 31),
            (34, 33),
            (34, 37),
            (34, 39),
            (35, 13),
            (35, 15),
            (35, 28),
            (36, 13),
            (36, 15),
            (36, 28),
            (37, 9),
            (37, 13),
            (37, 15),
            (37, 28),
            (37, 37),
            (38, 9),
            (38, 13),
            (38, 15),
            (38, 28),
            (39, 13),
        }:
            return 33
        return 10

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, num_mlp_0_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output):
        key = num_attn_1_2_output
        return 11

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_1_2_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_0_3_output, num_attn_2_3_output):
        key = (num_attn_0_3_output, num_attn_2_3_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
        }:
            return 19
        elif key in {
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (1, 6),
            (1, 7),
            (1, 8),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 0),
            (4, 1),
        }:
            return 10
        return 2

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_2_3_outputs)
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


print(run(["<s>", "0", "1", "3", "0", "0", "0", "5", "5", "3", "2", "3"]))
