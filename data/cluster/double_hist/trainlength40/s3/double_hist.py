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
        "output/length/rasp/double_hist/trainlength40/s3/double_hist_weights.csv",
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
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4", "<s>"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"1", "0", "4"}:
            return k_token == "3"
        elif q_token in {"3", "5", "2"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == "4"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
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

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"4", "3", "0", "2", "<s>"}:
            return k_token == "2"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"5"}:
            return k_token == "5"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 0
        elif q_position in {1, 2, 3, 4}:
            return k_position == 5
        elif q_position in {45, 5}:
            return k_position == 43
        elif q_position in {33, 37, 7, 46, 23}:
            return k_position == 46
        elif q_position in {39, 8, 14, 20, 24}:
            return k_position == 42
        elif q_position in {36, 9, 43, 48, 27}:
            return k_position == 47
        elif q_position in {10, 30}:
            return k_position == 40
        elif q_position in {34, 41, 42, 11, 15, 19, 22, 28}:
            return k_position == 44
        elif q_position in {32, 12, 18, 21, 31}:
            return k_position == 49
        elif q_position in {17, 35, 13, 47}:
            return k_position == 41
        elif q_position in {16}:
            return k_position == 45
        elif q_position in {38, 40, 44, 25, 26, 29}:
            return k_position == 48
        elif q_position in {49}:
            return k_position == 7

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 41
        elif q_position in {1, 6}:
            return k_position == 17
        elif q_position in {2, 3}:
            return k_position == 27
        elif q_position in {9, 4, 45}:
            return k_position == 18
        elif q_position in {43, 5, 46}:
            return k_position == 14
        elif q_position in {7}:
            return k_position == 31
        elif q_position in {8, 12}:
            return k_position == 32
        elif q_position in {10, 11}:
            return k_position == 35
        elif q_position in {13, 23}:
            return k_position == 36
        elif q_position in {38, 14}:
            return k_position == 34
        elif q_position in {15}:
            return k_position == 30
        elif q_position in {16, 34}:
            return k_position == 40
        elif q_position in {17, 21}:
            return k_position == 39
        elif q_position in {18}:
            return k_position == 25
        elif q_position in {33, 19, 35}:
            return k_position == 46
        elif q_position in {32, 20, 28}:
            return k_position == 48
        elif q_position in {22}:
            return k_position == 33
        elif q_position in {24}:
            return k_position == 44
        elif q_position in {25}:
            return k_position == 37
        elif q_position in {26}:
            return k_position == 43
        elif q_position in {27}:
            return k_position == 42
        elif q_position in {29}:
            return k_position == 38
        elif q_position in {49, 30}:
            return k_position == 45
        elif q_position in {31}:
            return k_position == 47
        elif q_position in {36}:
            return k_position == 49
        elif q_position in {37}:
            return k_position == 22
        elif q_position in {41, 39}:
            return k_position == 12
        elif q_position in {40}:
            return k_position == 15
        elif q_position in {42, 47}:
            return k_position == 9
        elif q_position in {44}:
            return k_position == 8
        elif q_position in {48}:
            return k_position == 11

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"4", "3", "1", "0", "5", "2"}:
            return k_token == ""
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 42
        elif q_position in {1, 42, 4, 28}:
            return k_position == 32
        elif q_position in {2, 18, 22, 23}:
            return k_position == 35
        elif q_position in {3}:
            return k_position == 38
        elif q_position in {9, 20, 5}:
            return k_position == 29
        elif q_position in {14, 6, 7}:
            return k_position == 24
        elif q_position in {8, 10, 11, 12}:
            return k_position == 25
        elif q_position in {43, 13}:
            return k_position == 26
        elif q_position in {24, 17, 21, 15}:
            return k_position == 33
        elif q_position in {16, 19}:
            return k_position == 27
        elif q_position in {25}:
            return k_position == 46
        elif q_position in {33, 35, 36, 26, 31}:
            return k_position == 37
        elif q_position in {40, 27, 47}:
            return k_position == 36
        elif q_position in {29}:
            return k_position == 9
        elif q_position in {30}:
            return k_position == 41
        elif q_position in {32}:
            return k_position == 21
        elif q_position in {34}:
            return k_position == 10
        elif q_position in {37}:
            return k_position == 7
        elif q_position in {38}:
            return k_position == 18
        elif q_position in {39}:
            return k_position == 49
        elif q_position in {41}:
            return k_position == 0
        elif q_position in {49, 44}:
            return k_position == 43
        elif q_position in {45}:
            return k_position == 30
        elif q_position in {46}:
            return k_position == 11
        elif q_position in {48}:
            return k_position == 13

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_3_output):
        key = (attn_0_1_output, attn_0_3_output)
        if key in {
            ("0", "<s>"),
            ("2", "1"),
            ("2", "2"),
            ("2", "<s>"),
            ("5", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 21
        return 3

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {
            ("2", "1"),
            ("2", "2"),
            ("2", "<s>"),
            ("5", "1"),
            ("5", "2"),
            ("5", "<s>"),
            ("<s>", "5"),
        }:
            return 1
        elif key in {("<s>", "1"), ("<s>", "2"), ("<s>", "<s>")}:
            return 21
        return 3

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 6

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 9

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_token, k_token):
        if q_token in {"5", "0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"<s>", "2"}:
            return k_token == "<s>"
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "0"

    attn_1_0_pattern = select_closest(tokens, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"0", "2"}:
            return k_token == "5"
        elif q_token in {"1", "3"}:
            return k_token == "0"
        elif q_token in {"4"}:
            return k_token == "1"
        elif q_token in {"5"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == "2"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, positions)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "3"
        elif attn_0_2_output in {"1", "5"}:
            return token == "4"
        elif attn_0_2_output in {"2"}:
            return token == "2"
        elif attn_0_2_output in {"3"}:
            return token == "0"
        elif attn_0_2_output in {"4"}:
            return token == "5"
        elif attn_0_2_output in {"<s>"}:
            return token == ""

    attn_1_2_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_1_output, position):
        if attn_0_1_output in {"1", "3", "0", "<s>"}:
            return position == 4
        elif attn_0_1_output in {"2"}:
            return position == 31
        elif attn_0_1_output in {"4", "5"}:
            return position == 6

    attn_1_3_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, positions)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_0_output, token):
        if attn_0_0_output in {"0"}:
            return token == "0"
        elif attn_0_0_output in {"1"}:
            return token == "1"
        elif attn_0_0_output in {"2"}:
            return token == "2"
        elif attn_0_0_output in {"3", "<s>"}:
            return token == "3"
        elif attn_0_0_output in {"4"}:
            return token == "4"
        elif attn_0_0_output in {"5"}:
            return token == "5"

    num_attn_1_0_pattern = select(tokens, attn_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"0"}:
            return attn_0_2_output == "0"
        elif attn_0_3_output in {"1"}:
            return attn_0_2_output == "1"
        elif attn_0_3_output in {"<s>", "2"}:
            return attn_0_2_output == ""
        elif attn_0_3_output in {"3"}:
            return attn_0_2_output == "3"
        elif attn_0_3_output in {"4"}:
            return attn_0_2_output == "4"
        elif attn_0_3_output in {"5"}:
            return attn_0_2_output == "5"

    num_attn_1_1_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, mlp_0_1_output):
        if attn_0_1_output in {"0"}:
            return mlp_0_1_output == 35
        elif attn_0_1_output in {"1", "<s>"}:
            return mlp_0_1_output == 48
        elif attn_0_1_output in {"2"}:
            return mlp_0_1_output == 37
        elif attn_0_1_output in {"3"}:
            return mlp_0_1_output == 41
        elif attn_0_1_output in {"4"}:
            return mlp_0_1_output == 2
        elif attn_0_1_output in {"5"}:
            return mlp_0_1_output == 18

    num_attn_1_2_pattern = select(mlp_0_1_outputs, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(token, mlp_0_1_output):
        if token in {"1", "5", "0"}:
            return mlp_0_1_output == 38
        elif token in {"2"}:
            return mlp_0_1_output == 21
        elif token in {"3"}:
            return mlp_0_1_output == 20
        elif token in {"4"}:
            return mlp_0_1_output == 27
        elif token in {"<s>"}:
            return mlp_0_1_output == 8

    num_attn_1_3_pattern = select(mlp_0_1_outputs, tokens, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_1_output):
        key = (position, attn_1_1_output)
        if key in {
            (0, 1),
            (0, 16),
            (0, 36),
            (0, 39),
            (2, 1),
            (2, 16),
            (2, 36),
            (2, 39),
            (3, 1),
            (3, 16),
            (3, 36),
            (3, 39),
            (5, 36),
            (7, 1),
            (7, 16),
            (7, 36),
            (7, 39),
            (8, 1),
            (8, 16),
            (8, 36),
            (8, 39),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 11),
            (9, 13),
            (9, 14),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 24),
            (9, 25),
            (9, 26),
            (9, 27),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
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
            (11, 36),
            (12, 1),
            (12, 16),
            (12, 36),
            (12, 39),
            (15, 1),
            (15, 16),
            (15, 36),
            (15, 39),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 8),
            (18, 11),
            (18, 13),
            (18, 14),
            (18, 16),
            (18, 17),
            (18, 18),
            (18, 21),
            (18, 22),
            (18, 25),
            (18, 26),
            (18, 27),
            (18, 28),
            (18, 29),
            (18, 30),
            (18, 31),
            (18, 33),
            (18, 35),
            (18, 36),
            (18, 37),
            (18, 38),
            (18, 39),
            (18, 40),
            (18, 41),
            (18, 42),
            (18, 43),
            (18, 44),
            (18, 45),
            (18, 46),
            (18, 47),
            (18, 48),
            (18, 49),
            (19, 1),
            (19, 16),
            (19, 36),
            (19, 39),
            (23, 1),
            (23, 16),
            (23, 26),
            (23, 36),
            (23, 38),
            (23, 39),
            (23, 43),
            (25, 1),
            (25, 16),
            (25, 36),
            (25, 39),
            (27, 16),
            (27, 36),
            (29, 1),
            (29, 16),
            (29, 36),
            (29, 39),
            (30, 1),
            (30, 16),
            (30, 36),
            (30, 39),
            (32, 36),
            (33, 1),
            (33, 16),
            (33, 36),
            (33, 39),
            (34, 1),
            (34, 16),
            (34, 36),
            (34, 39),
            (35, 1),
            (35, 16),
            (35, 36),
            (35, 39),
            (36, 16),
            (36, 36),
            (37, 16),
            (37, 36),
            (38, 1),
            (38, 16),
            (38, 36),
            (38, 39),
            (39, 1),
            (39, 8),
            (39, 16),
            (39, 26),
            (39, 30),
            (39, 33),
            (39, 36),
            (39, 38),
            (39, 39),
            (39, 45),
            (39, 48),
            (39, 49),
            (40, 16),
            (40, 36),
            (42, 1),
            (42, 16),
            (42, 36),
            (42, 39),
            (43, 1),
            (43, 8),
            (43, 16),
            (43, 26),
            (43, 36),
            (43, 38),
            (43, 39),
            (43, 43),
            (43, 45),
            (43, 49),
            (44, 1),
            (44, 16),
            (44, 36),
            (44, 39),
            (45, 1),
            (45, 16),
            (45, 36),
            (45, 39),
            (46, 1),
            (46, 16),
            (46, 26),
            (46, 36),
            (46, 38),
            (46, 39),
            (46, 43),
            (46, 45),
            (46, 49),
            (47, 1),
            (47, 16),
            (47, 36),
            (47, 39),
            (48, 36),
            (49, 1),
            (49, 16),
            (49, 36),
            (49, 39),
        }:
            return 10
        elif key in {(38, 23), (38, 24), (38, 34)}:
            return 9
        return 0

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_1_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(token, attn_1_0_output):
        key = (token, attn_1_0_output)
        if key in {
            ("0", 13),
            ("0", 34),
            ("0", 49),
            ("2", 13),
            ("2", 34),
            ("2", 49),
            ("<s>", 13),
            ("<s>", 34),
        }:
            return 10
        return 0

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(tokens, attn_1_0_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_0_3_output):
        key = (num_attn_1_2_output, num_attn_0_3_output)
        return 4

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output):
        key = num_attn_1_2_output
        return 38

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_1_2_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "4"
        elif attn_0_1_output in {"4", "3", "1", "5", "<s>", "2"}:
            return token == ""

    attn_2_0_pattern = select_closest(tokens, attn_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, num_mlp_0_0_output):
        if token in {"0"}:
            return num_mlp_0_0_output == 4
        elif token in {"1", "3"}:
            return num_mlp_0_0_output == 17
        elif token in {"2"}:
            return num_mlp_0_0_output == 26
        elif token in {"4"}:
            return num_mlp_0_0_output == 36
        elif token in {"5"}:
            return num_mlp_0_0_output == 18
        elif token in {"<s>"}:
            return num_mlp_0_0_output == 0

    attn_2_1_pattern = select_closest(num_mlp_0_0_outputs, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"4", "5", "0", "2"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "4"
        elif q_token in {"3"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "4"
        elif attn_0_2_output in {"1", "3", "2"}:
            return token == "2"
        elif attn_0_2_output in {"4"}:
            return token == "1"
        elif attn_0_2_output in {"5", "<s>"}:
            return token == ""

    attn_2_3_pattern = select_closest(tokens, attn_0_2_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"4", "3", "1", "0", "5", "<s>", "2"}:
            return k_attn_0_1_output == ""

    num_attn_2_0_pattern = select(attn_0_1_outputs, attn_0_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_0_output, num_mlp_1_0_output):
        if attn_1_0_output in {0, 16, 31}:
            return num_mlp_1_0_output == 43
        elif attn_1_0_output in {1, 36, 5}:
            return num_mlp_1_0_output == 25
        elif attn_1_0_output in {2}:
            return num_mlp_1_0_output == 26
        elif attn_1_0_output in {3}:
            return num_mlp_1_0_output == 40
        elif attn_1_0_output in {32, 25, 4}:
            return num_mlp_1_0_output == 15
        elif attn_1_0_output in {27, 6, 39}:
            return num_mlp_1_0_output == 39
        elif attn_1_0_output in {33, 7}:
            return num_mlp_1_0_output == 42
        elif attn_1_0_output in {8, 17, 42, 14}:
            return num_mlp_1_0_output == 10
        elif attn_1_0_output in {9, 41}:
            return num_mlp_1_0_output == 11
        elif attn_1_0_output in {48, 10}:
            return num_mlp_1_0_output == 7
        elif attn_1_0_output in {11, 28}:
            return num_mlp_1_0_output == 48
        elif attn_1_0_output in {40, 12}:
            return num_mlp_1_0_output == 46
        elif attn_1_0_output in {37, 13}:
            return num_mlp_1_0_output == 13
        elif attn_1_0_output in {15}:
            return num_mlp_1_0_output == 14
        elif attn_1_0_output in {18, 21}:
            return num_mlp_1_0_output == 18
        elif attn_1_0_output in {19}:
            return num_mlp_1_0_output == 19
        elif attn_1_0_output in {49, 20}:
            return num_mlp_1_0_output == 12
        elif attn_1_0_output in {22}:
            return num_mlp_1_0_output == 22
        elif attn_1_0_output in {23}:
            return num_mlp_1_0_output == 28
        elif attn_1_0_output in {24}:
            return num_mlp_1_0_output == 8
        elif attn_1_0_output in {26, 44}:
            return num_mlp_1_0_output == 4
        elif attn_1_0_output in {29}:
            return num_mlp_1_0_output == 41
        elif attn_1_0_output in {35, 38, 43, 46, 30}:
            return num_mlp_1_0_output == 36
        elif attn_1_0_output in {34}:
            return num_mlp_1_0_output == 1
        elif attn_1_0_output in {45}:
            return num_mlp_1_0_output == 45
        elif attn_1_0_output in {47}:
            return num_mlp_1_0_output == 6

    num_attn_2_1_pattern = select(
        num_mlp_1_0_outputs, attn_1_0_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_1_output, attn_1_2_output):
        if attn_0_1_output in {"0"}:
            return attn_1_2_output == 31
        elif attn_0_1_output in {"1", "3"}:
            return attn_1_2_output == 37
        elif attn_0_1_output in {"2"}:
            return attn_1_2_output == 21
        elif attn_0_1_output in {"4"}:
            return attn_1_2_output == 38
        elif attn_0_1_output in {"5"}:
            return attn_1_2_output == 30
        elif attn_0_1_output in {"<s>"}:
            return attn_1_2_output == 0

    num_attn_2_2_pattern = select(attn_1_2_outputs, attn_0_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_0_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_3_output, position):
        if attn_0_3_output in {"0"}:
            return position == 13
        elif attn_0_3_output in {"1"}:
            return position == 30
        elif attn_0_3_output in {"2"}:
            return position == 37
        elif attn_0_3_output in {"3"}:
            return position == 12
        elif attn_0_3_output in {"4"}:
            return position == 16
        elif attn_0_3_output in {"5"}:
            return position == 19
        elif attn_0_3_output in {"<s>"}:
            return position == 47

    num_attn_2_3_pattern = select(positions, attn_0_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_3_output, attn_1_2_output):
        key = (attn_1_3_output, attn_1_2_output)
        if key in {
            (0, 17),
            (0, 38),
            (0, 42),
            (0, 47),
            (10, 17),
            (10, 42),
            (25, 17),
            (25, 42),
            (34, 17),
            (34, 42),
            (38, 17),
            (38, 38),
            (38, 42),
            (42, 17),
            (42, 42),
            (44, 17),
            (44, 42),
        }:
            return 44
        return 34

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_2_output, attn_2_1_output):
        key = (attn_2_2_output, attn_2_1_output)
        return 19

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_2_1_output):
        key = (num_attn_1_2_output, num_attn_2_1_output)
        return 6

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_2_output, num_attn_1_2_output):
        key = (num_attn_2_2_output, num_attn_1_2_output)
        return 41

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_2_outputs)
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


print(
    run(
        [
            "<s>",
            "1",
            "3",
            "0",
            "0",
            "0",
            "5",
            "5",
            "3",
            "2",
            "3",
            "1",
            "1",
            "2",
            "5",
            "0",
            "4",
            "4",
            "5",
            "0",
            "2",
            "1",
            "2",
            "2",
            "2",
            "4",
        ]
    )
)
