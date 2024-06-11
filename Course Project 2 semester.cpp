#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>

using namespace std;

typedef long long ll;
typedef long double ld;
const double EULER = 2.71828182845904523536;

pair<ld, ld> lin_reg_2dim(vector<pair<ld, ld>> obs, int n) { // базовый вариант
    ld x_i = 0;
    ld y_i = 0;
    ld x_sq_i = 0;
    ld x_y_i = 0;
    for (int i = 0; i < n; i++) {
        x_i += obs[i].first;
        y_i += obs[i].second;
        x_sq_i += pow(obs[i].first, 2);
        x_y_i += obs[i].first * obs[i].second;
    }

    ld det = x_sq_i * n - x_i * x_i;
    ld deta = x_y_i * n - x_i * y_i;
    ld detb = x_sq_i * y_i - x_y_i * x_i;

    ld a = deta / det;
    ld b = detb / det;
    return { a, b };
}

ld lin_reg_2dim_center(vector<pair<ld, ld>> obs, int n) { // центрирует данные, возвращает a по упрощенным формулам
    ld xm = 0;
    ld ym = 0;
    for (int i = 0; i < n; i++) {
        xm += obs[i].first;
        ym += obs[i].second;
    }
    xm /= n;
    ym /= n;

    ld xyc_i = 0;
    ld xxc_i = 0;
    for (int i = 0; i < n; i++) {
        xyc_i += (obs[i].first - xm) * (obs[i].second - ym);
        xxc_i += (obs[i].first - xm) * (obs[i].first - xm);
    }

    return xyc_i / xxc_i;
}

vector<vector<ld>> mul(vector<vector<ld>> a, vector<vector<ld>> b) { // вспомогательные функции для матриц
    int n = a.size();
    int p = a[0].size();
    int m = b[0].size();
    vector<vector<ld>> c(n, vector<ld>(m, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < p; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

vector<vector<ld>> tr(vector<vector<ld>> a) {
    int n = a.size();
    int m = a[0].size();
    vector<vector<ld>> b(m, vector<ld>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[j][i] = a[i][j];
        }
    }
    return b;
}

vector<vector<ld>> minor(vector<vector<ld>> a, int k, int l) {
    int n = a.size() - 1;
    vector<vector<ld>> b(n, vector<ld>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int t = i;
            int s = j;
            if (i >= k) {
                t++;
            }
            if (j >= l) {
                s++;
            }
            b[i][j] = a[t][s];
        }
    }
    return b;
}

ld det(vector<vector<ld>> a) {
    int n = a.size();
    if (n == 1) {
        return a[0][0];
    }
    ld dt = 0;
    for (int j = 0; j < n; j++) {
        dt += (j % 2 == 0 ? 1 : -1) * a[0][j] * det(minor(a, 0, j));
    }
    return dt;
}

vector<vector<ld>> inv(vector<vector<ld>> a) {
    int n = a.size();
    ld dt = det(a);
    vector<vector<ld>> adj(n, vector<ld>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj[i][j] = ((i + j) % 2 == 0 ? 1 : -1) * det(minor(a, j, i)) / dt;
        }
    }
    return adj;
}

pair<ld, ld> lin_reg_2dim_matrix(vector<pair<ld, ld>> obs, int n) { // считает коэффициенты через матрицы в линейном тренде
    vector<vector<ld>> x(n, vector<ld>(2, 0));
    vector<vector<ld>> y(n, vector<ld>(1, 0));
    for (int i = 0; i < n; i++) {
        x[i][0] = 1;
        x[i][1] = obs[i].first;
        y[i][0] = obs[i].second;
    }
    vector<vector<ld>> ans = mul(mul(inv(mul(tr(x), x)), tr(x)), y);
    return { ans[1][0], ans[0][0] };
}

vector<ld> lin_reg_sq(vector<pair<ld, ld>> obs, int n) { // считатет коэффициенты в квадратичном тренде
    ld xm = 0;
    ld xsq_m = 0;
    ld ym = 0;
    for (int i = 0; i < n; i++) {
        xm += obs[i].first;
        xsq_m += obs[i].first * obs[i].first;
        ym += obs[i].second;
    }
    xm /= n;
    xsq_m /= n;
    ym /= n;
    vector<vector<ld>> X(n, vector<ld>(2, 0));
    vector<vector<ld>> Y(n, vector<ld>(1, 0));
    for (int i = 0; i < n; i++) {
        X[i][0] = obs[i].first - xm;
        X[i][1] = obs[i].first * obs[i].first - xsq_m;
        Y[i][0] = obs[i].second - ym;
    }
    vector<vector<ld>> ans = mul(mul(inv(mul(tr(X), X)), tr(X)), Y);

    return { ans[1][0], ans[0][0] };
}

vector<ld> lin_reg_polynom(vector<pair<ld, ld>> obs, int deg, int n) { // полиномиальный тренд
    vector<vector<ld>> X(n, vector<ld>(deg + 1, 0));
    vector<vector<ld>> Y(n, vector<ld>(1, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= deg; j++) {
            X[i][j] = pow(obs[i].first, j);
        }
        Y[i][0] = obs[i].second;
    }

    vector<vector<ld>> ans = mul(mul(inv(mul(tr(X), X)), tr(X)), Y);
    vector<ld> coef(deg + 1, 0);
    for (int i = 0; i <= deg; i++) {
        coef[i] = ans[deg - i][0];
    }
    return coef;
}

vector<pair<ld, ld>> predict(vector<ld> x, vector<ld> coef) {
    vector<pair<ld, ld>> pred;
    for (int i = 0; i < x.size(); i++) {
        ld y = 0;
        for (int j = 0; j < coef.size(); j++) {
            y += coef[j] * pow(x[i], coef.size() - j - 1);
        }
        pred.push_back({ x[i], y });
    }
    return pred;
}

ld mse(vector<pair<ld, ld>> target, vector<pair<ld, ld>> pred) {
    ld mse = 0;
    for (int i = 0; i < target.size(); i++) {
        mse += pow(abs(target[i].second - pred[i].second), 2);
    }
    mse /= target.size();
    return mse;
}

vector<pair<ld, ld>> lin_f(ld a, ld b, ld bound, int n, default_random_engine& re, uniform_real_distribution<ld>& unif) { // линейные данные с шумом
    vector<pair<ld, ld>> obs;
    for (int x_i = 1; x_i <= n; x_i++) {
        obs.push_back({ x_i, a * x_i + b + unif(re)});
    }
    return obs;
}

vector<pair<ld, ld>> sq_f(ld a, ld b, ld c, ld bound, int n, default_random_engine& re, uniform_real_distribution<ld>& unif) { // квадратичные данные с шумом
    vector<pair<ld, ld>> obs;
    for (int x_i = 1; x_i <= n; x_i++) {
        obs.push_back({ x_i, a * x_i * x_i + b * x_i + c + unif(re) });
    }
    return obs;
}

int main()
{
    default_random_engine re(time(NULL));
    ifstream fin;
    fin.open("adv72200.txt");
    ofstream fout;
    fout.open("obs.txt");

    string s;
    for (int i = 0; i < 5; i++) {
        getline(fin, s);
    }
    vector<pair<ld, ld>> obs;
    int t = 0;
    for (int i = 0; i < 30 * 12 + 4; i++, t++) {
        int y;
        fin >> y;
        if (i % 13 == 0) {
            t--;
            continue;
        }
        obs.push_back({ t, y });
    }
    vector<pair<ld, ld>> train;
    vector<pair<ld, ld>> valid;
    vector<pair<ld, ld>> test;
    vector<ld> x_v;
    vector<ld> x_t;
    vector<ld> x_train;

    for (int i = 0; i < obs.size(); i++) {
        if (i <= 21 * 12) {
            train.push_back({ obs[i].first, log(obs[i].second) });
            x_train.push_back(i);
        }
        else if (i > 21 * 12 && i <= 24 * 12) {
            valid.push_back({ obs[i].first, obs[i].second });
            x_v.push_back(i);
        }
        if (i > 24 * 12){
            test.push_back({ obs[i].first, obs[i].second });
            x_t.push_back(i);
        }
    }
    for (int i = 1; i <= 4; i++) {
        vector<ld> coef = lin_reg_polynom(train, i, train.size());
        for (int j = 0; j < coef.size(); j++) {
            fout << coef[j] << " ";
        }
        fout << endl;
        vector<pair<ld, ld>> pred_train = predict(x_train, coef);
        for (int j = 0; j < pred_train.size(); j++) {
            pred_train[j] = { pred_train[j].first, pow(EULER, pred_train[j].second) };
            train[j] = { train[j].first, pow(EULER, train[j].second) };
        }
        fout << scientific << (mse(train, pred_train)) << " ";
        for (int j = 0; j < pred_train.size(); j++) {
            train[j] = { train[j].first, log(train[j].second) };
        }
        vector<pair<ld, ld>> pred_valid = predict(x_v, coef);
        for (int j = 0; j < pred_valid.size(); j++) {
            pred_valid[j] = { pred_valid[j].first, pow(EULER, pred_valid[j].second) };
        }
        fout << scientific << (mse(valid, pred_valid)) << " ";
        vector<pair<ld, ld>> pred_test = predict(x_t, coef);
        for (int j = 0; j < pred_test.size(); j++) {
            pred_test[j] = { pred_test[j].first, pow(EULER, pred_test[j].second) };
        }
        fout << scientific << mse(test, pred_test) << endl << endl;
    } 
    fout.close();
}
