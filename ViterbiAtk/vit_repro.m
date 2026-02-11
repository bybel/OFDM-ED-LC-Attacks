clear; close all; clc;

%% --- 1. Scenario Selector ---
% Select the scenario to reproduce specific slide results
% 'Slide20': 16QAM, 5 Taps, Tob=0.50   (Warning: 64k States - Very Slow)
% 'Slide21': 16QAM, 5 Taps, Tob=0.5625 (Warning: 64k States - Very Slow)
% 'Slide22': 16QAM, 3 Taps, Tob=0.5625 (256 States - Fast)
% 'Slide23': 16QAM, 3 Taps, Tob=0.75   (256 States - Fast)
% 'Slide24': 64QAM, 3 Taps, Tob=0.75   (4096 States - Medium)
% 'Slide25': 256QAM, 3 Taps, Tob=0.75  (65k States - Very Slow)

SCENARIO = 'Slide25'; 

switch SCENARIO
    case 'Slide20'
        Modulation = '16QAM'; N_taps = 5; T_ob = 1/2;
    case 'Slide21'
        Modulation = '16QAM'; N_taps = 5; T_ob = 9/16;
    case 'Slide22'
        Modulation = '16QAM'; N_taps = 3; T_ob = 9/16;
    case 'Slide23'
        Modulation = '16QAM'; N_taps = 3; T_ob = 3/4;
    case 'Slide24'
        Modulation = '64QAM'; N_taps = 3; T_ob = 3/4;
    case 'Slide25'
        Modulation = '256QAM'; N_taps = 3; T_ob = 3/4;
    otherwise
        error('Unknown Scenario');
end

% Simulation Parameters
N_fft = 128;
N_subcarriers = 122;
SNR_dB = 30;
N_trials = 50; % Adjust based on performance (50 is good for 3-tap)

fprintf('Running %s: %s, %d Taps, Tob=%.4f\n', SCENARIO, Modulation, N_taps, T_ob);

%% --- 2. Setup Constants & Viterbi Trellis ---

% A. Modulation Parameters
switch Modulation
    case '16QAM',  bps = 4;
    case '64QAM',  bps = 6;
    case '256QAM', bps = 8;
end
M = 2^bps;
N_states = M^(N_taps - 1);
fprintf('Viterbi Complexity: %d States per subcarrier\n', N_states);

% B. Channel / ICI Coefficients extraction
n_ob_samples = floor(N_fft * T_ob);
win = zeros(N_fft, 1);
win(1:n_ob_samples) = hamming(n_ob_samples); 

% FFT of window (Circular convolution kernel)
W_freq = fft(win) / N_fft; 

% Extract Taps centered at DC (Index 1)
if N_taps == 3
    tap_indices = [N_fft, 1, 2];
elseif N_taps == 5
    tap_indices = [N_fft-1, N_fft, 1, 2, 3];
else
    error('Only 3 or 5 taps supported');
end
h_ici = W_freq(tap_indices).'; 

% C. Pre-compute Trellis Outputs
% State definition: (sym_{k-(Taps-1)+1}, ..., sym_{k})
sym_mapping = qammod(0:M-1, M, 'gray', 'UnitAveragePower', true);
Expected_Y = zeros(N_states, M); 
Next_State_Table = zeros(N_states, M);

fprintf('Building Trellis...');
for s = 0:N_states-1
    % Decode state into symbols
    state_syms = zeros(1, N_taps-1);
    temp_s = s;
    for i = 1:N_taps-1
        idx = mod(temp_s, M);
        state_syms(i) = idx; 
        temp_s = floor(temp_s / M);
    end
    
    for input_idx = 0:M-1
        % Calculate Expected Output
        val_vec = sym_mapping([input_idx, state_syms] + 1);
        Expected_Y(s+1, input_idx+1) = sum(val_vec .* h_ici);
        
        % Calculate Next State
        new_state_vec = [input_idx, state_syms(1:end-1)];
        ns = 0;
        for i = (N_taps-1):-1:1
            ns = ns * M + new_state_vec(i);
        end
        Next_State_Table(s+1, input_idx+1) = ns;
    end
end
fprintf(' Done.\n');

%% --- 3. Main Simulation Loop ---
ber_stats = zeros(N_trials, 1);
corr_stats = zeros(N_trials, 1);
sc_idx = [2:62, 68:128]; % Active Subcarriers

tic;
for trial = 1:N_trials
    if mod(trial, 10)==0, fprintf('Trial %d/%d\n', trial, N_trials); end
    
    % A. Transmitter
    n_bits_total = N_subcarriers * bps;
    tx_bits = randi([0 1], n_bits_total, 1);
    tx_syms_idx = bi2de(reshape(tx_bits, bps, [])', 'left-msb');
    tx_syms = sym_mapping(tx_syms_idx + 1).';
    
    X_tx = zeros(N_fft, 1);
    X_tx(sc_idx) = tx_syms;
    x_tx = ifft(X_tx, N_fft);
    
    % B. Channel (AWGN)
    sig_pwr = mean(abs(x_tx).^2);
    noise_pwr = sig_pwr / 10^(SNR_dB/10);
    noise = sqrt(noise_pwr/2) * (randn(size(x_tx)) + 1j*randn(size(x_tx)));
    x_rx = x_tx + noise;
    
    % C. Attacker (Partial Obs + Window + FFT)
    x_obs = x_rx;
    x_obs(1:n_ob_samples) = x_obs(1:n_ob_samples) .* hamming(n_ob_samples);
    x_obs(n_ob_samples+1:end) = 0; 
    
    Y_obs = fft(x_obs, N_fft);
    
    % D. Viterbi Decoder 
    path_metrics = inf(N_states, 1);
    path_metrics(1) = 0; 
    
    % Traceback matrix
    if N_states <= 65536
        tb_prev_state = zeros(length(sc_idx), N_states, 'uint16');
    else
        tb_prev_state = zeros(length(sc_idx), N_states, 'uint32');
    end
    
    % Forward Pass
    for k = 1:length(sc_idx)
        obs = Y_obs(sc_idx(k));
        branch_metrics = abs(obs - Expected_Y).^2;
        total_metrics = path_metrics + branch_metrics; 
        
        new_metrics = inf(N_states, 1);
        
        for s = 1:N_states
            for input_idx = 1:M
                ns = Next_State_Table(s, input_idx) + 1; 
                cost = total_metrics(s, input_idx);
                
                if cost < new_metrics(ns)
                    new_metrics(ns) = cost;
                    tb_prev_state(k, ns) = s; 
                end
            end
        end
        path_metrics = new_metrics;
    end
    
    % Traceback
    [~, best_end_state] = min(path_metrics);
    state_seq = zeros(length(sc_idx)+1, 1);
    state_seq(end) = best_end_state;
    for k = length(sc_idx):-1:1
        state_seq(k) = tb_prev_state(k, state_seq(k+1));
    end
    
    % Recover Inputs
    rec_inputs = zeros(length(sc_idx), 1);
    for k = 1:length(sc_idx)
        prev = state_seq(k) - 1;
        curr = state_seq(k+1) - 1;
        found = find(Next_State_Table(prev+1, :) == curr, 1);
        rec_inputs(k) = found - 1; 
    end
    
    % Sync Check (BER)
    rx_bits_mat = de2bi(rec_inputs, bps, 'left-msb')';
    rx_bits = rx_bits_mat(:);
    
    ber_current = 0.5;
    for shift = -2:2
        tx_shifted = circshift(tx_bits, shift*bps);
        errs = sum(rx_bits ~= tx_shifted) / n_bits_total;
        if errs < ber_current, ber_current = errs; end
    end
    ber_stats(trial) = ber_current;
    
    % E. Correlation Calc
    rx_syms_rec = sym_mapping(rec_inputs + 1).';
    X_atk_rec = zeros(N_fft, 1);
    X_atk_rec(sc_idx) = rx_syms_rec;
    x_atk = ifft(X_atk_rec, N_fft);
    
    tail_true = x_tx(n_ob_samples+1:end);
    tail_atk = x_atk(n_ob_samples+1:end);
    
    num = abs(tail_atk' * tail_true);
    den = norm(tail_atk) * norm(tail_true);
    corr_stats(trial) = num / max(den, 1e-9);
end
toc;

%% --- 4. Plotting ---
figure('Position', [100,100,1000,400], 'Name', SCENARIO);
subplot(1,2,1);
[f,x] = ecdf(ber_stats);
plot(x, f, 'b-', 'LineWidth', 2);
xlabel('BER'); ylabel('CDF'); title([SCENARIO ' BER']); grid on; xlim([0 0.5]);

subplot(1,2,2);
[f,x] = ecdf(corr_stats);
plot(x, f, 'r-', 'LineWidth', 2);
xlabel('Correlation'); ylabel('CDF'); title([SCENARIO ' Correlation']); grid on; xlim([0 1]);

fprintf('Result %s: Mean BER %.4f, Mean Corr %.4f\n', SCENARIO, mean(ber_stats), mean(corr_stats));