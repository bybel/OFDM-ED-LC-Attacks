clear; close all; clc;

%% Parameters 
N_fft = 128;         % FFT size
N_subcarriers = 122; % Active subcarriers
SNR_dB = 30;         % SNR
T_ob = 5/8;          % Observation window (0.625)
N_attack_trials = 10000; % Number of Packet trials
N_symbols_per_packet = 1; % Single symbol (matches Slide 9 variance)

% Derived
n_ob_samples = round(N_fft * T_ob);
n_atk_samples = N_fft - n_ob_samples;
slicer_scaling_factor = T_ob; 

%% Modulations (Gray Mapping)
modulations = {'256QAM', '64QAM', '16QAM', '8PSK', 'QPSK'};
bits_per_symbol = [8, 6, 4, 3, 2];
colors = {'b', 'g', 'r', 'c', [0.5 0 0.5]}; 

%% Run simulations
fprintf('Running simulations (Gray, Corr, Packet Avg=%d): N=%d, Tob=%.3f, SNR=%d dB\n', ...
    N_symbols_per_packet, N_fft, T_ob, SNR_dB);

RESULTS = struct();

% AWGN Baseline for Correlation
awgn_corr_values = zeros(N_attack_trials, 1);

for mod_idx = 1:length(modulations)
    mod_type = modulations{mod_idx};
    fprintf('Simulating %s...\n', mod_type);
    
    bps = bits_per_symbol(mod_idx);
    M = 2^bps;
    
    ber_values = zeros(N_attack_trials, 1);
    corr_values = zeros(N_attack_trials, 1);
    
    for trial = 1:N_attack_trials
        
        % --- 1. Transmission ---
        n_total_bits = N_subcarriers * bps * N_symbols_per_packet;
        tx_bits = randi([0 1], n_total_bits, 1);
        
        tx_bits_mat = reshape(tx_bits, bps, [])';
        tx_syms_idx = bi2de(tx_bits_mat, 'left-msb');
        
        % Gray Mapping & Unit Power
        if contains(mod_type, 'QPSK')
            tx_syms = pskmod(tx_syms_idx, M, pi/4, 'gray');
        elseif contains(mod_type, 'PSK')
             tx_syms = pskmod(tx_syms_idx, M, 0, 'gray');
        else
             tx_syms = qammod(tx_syms_idx, M, 'gray', 'UnitAveragePower', true);
        end
        
        % Map to Grid [N_subcarriers x N_syms]
        tx_syms_grid = reshape(tx_syms, N_subcarriers, N_symbols_per_packet);
        
        X_tx = zeros(N_fft, N_symbols_per_packet);
        X_tx(2:62, :) = tx_syms_grid(1:61, :);
        X_tx(68:128, :) = tx_syms_grid(62:122, :);
        
        x_tx = ifft(X_tx, N_fft);
        
        % --- 2. Channel ---
        sig_pwr = mean(mean(abs(x_tx).^2));
        noise_pwr = sig_pwr / (10^(SNR_dB/10));
        noise = sqrt(noise_pwr/2) * (randn(size(x_tx)) + 1j*randn(size(x_tx)));
        x_rx = x_tx + noise;
        
        % --- 3. Attack (Slicing) ---
        x_part = x_rx(1:n_ob_samples, :);
        
        % Zero Pad & FFT
        x_pad = [x_part; zeros(n_atk_samples, N_symbols_per_packet)];
        X_est = fft(x_pad, N_fft);
        
        % Extract & Scale
        X_rec_data = zeros(N_subcarriers, N_symbols_per_packet);
        X_rec_data(1:61, :) = X_est(2:62, :);
        X_rec_data(62:122, :) = X_est(68:128, :);
        
        Y_rec = X_rec_data / slicer_scaling_factor;
        
        % Hard Decision (The "Slicing")
        Y_rec_serial = Y_rec(:);
        if contains(mod_type, 'QPSK')
            rx_bits = pskdemod(Y_rec_serial, M, pi/4, 'gray', 'OutputType', 'bit');
            sl_syms_idx = pskdemod(Y_rec_serial, M, pi/4, 'gray');
            sl_syms = pskmod(sl_syms_idx, M, pi/4, 'gray'); 
        elseif contains(mod_type, 'PSK')
             rx_bits = pskdemod(Y_rec_serial, M, 0, 'gray', 'OutputType', 'bit');
             sl_syms_idx = pskdemod(Y_rec_serial, M, 0, 'gray');
             sl_syms = pskmod(sl_syms_idx, M, 0, 'gray');
        else
             rx_bits = qamdemod(Y_rec_serial, M, 'gray', 'UnitAveragePower', true, 'OutputType', 'bit');
             sl_syms_idx = qamdemod(Y_rec_serial, M, 'gray', 'UnitAveragePower', true);
             sl_syms = qammod(sl_syms_idx, M, 'gray', 'UnitAveragePower', true);
        end
        
        % --- 4. BER Calc ---
        bit_errors = sum(rx_bits ~= tx_bits);
        ber_values(trial) = bit_errors / n_total_bits;
        
        % --- 5. Reconstruction & Correlation ---
        % Attacker regenerates the FULL symbol from sliced symbols
        sl_syms_grid = reshape(sl_syms, N_subcarriers, N_symbols_per_packet);
        
        X_atk = zeros(N_fft, N_symbols_per_packet);
        X_atk(2:62, :) = sl_syms_grid(1:61, :);
        X_atk(68:128, :) = sl_syms_grid(62:122, :);
        
        x_atk = ifft(X_atk, N_fft);
        
        % Correlation over Attack Period (Unobserved Tail)
        % We calculate correlation on the part of the signal that was NOT
        % observed (prediction), which is physically where the attack occurs.
        
        tail_tx = x_tx(n_ob_samples+1:end, :);
        tail_atk = x_atk(n_ob_samples+1:end, :);
        
        % Flatten to single vector for "Packet" correlation
        tail_tx_vec = tail_tx(:);
        tail_atk_vec = tail_atk(:);
        
        % Correlation Coefficient formula: |a'*b| / (norm(a)*norm(b))
        num = abs(tail_atk_vec' * tail_tx_vec);
        den = norm(tail_atk_vec) * norm(tail_tx_vec);
        
        if den == 0, den=1e-9; end
        corr_values(trial) = num / den;
        
        % Generate AWGN Baseline (One time per trial loop usually sufficient)
        if mod_idx == 1
            % Random noise as 'attack' signal
            tail_noise = randn(size(tail_tx_vec)) + 1j*randn(size(tail_tx_vec));
  
            num_n = abs(tail_noise' * tail_tx_vec);
            den_n = norm(tail_noise) * norm(tail_tx_vec);
            awgn_corr_values(trial) = num_n / den_n;
        end
    end
    
    % Sanitize field name (must start with letter)
    if isstrprop(mod_type(1), 'digit')
        field_name = ['M', mod_type];
    else
        field_name = mod_type;
    end
    
    RESULTS.(field_name).BER = ber_values;
    RESULTS.(field_name).Corr = corr_values;
    
    fprintf('  Mean BER: %.4f | Mean Corr: %.4f\n', mean(ber_values), mean(corr_values));
end

%% Plotting
% Figure 1: BER CDF
figure('Name', 'BER CDF');
hold on; grid on;
for mod_idx = 1:length(modulations)
    mod_type = modulations{mod_idx};
    if isstrprop(mod_type(1), 'digit'), f_name=['M', mod_type]; else, f_name=mod_type; end
    
    [sorted_ber, ~] = sort(RESULTS.(f_name).BER);
    cdf_v = (1:length(sorted_ber))' / length(sorted_ber);
    
    % Label
    bits = N_subcarriers * bits_per_symbol(mod_idx);
    lbl = sprintf('%s(%dbits)', mod_type, bits);
    
    plot(sorted_ber, cdf_v, 'LineWidth', 2, 'Color', colors{mod_idx}, 'DisplayName', lbl);
end
xlabel('Bit Error Rate'); ylabel('CDF');
title(sprintf('BER CDF (Gray, Avg=%d)', N_symbols_per_packet));
legend('Location', 'southeast');
xlim([0 0.5]); ylim([0 1]);

% Figure 2: Correlation CDF
figure('Name', 'Correlation CDF');
hold on; grid on;

% Plot AWGN First
[sorted_awgn, ~] = sort(awgn_corr_values);
cdf_awgn = (1:length(sorted_awgn))' / length(sorted_awgn);
plot(sorted_awgn, cdf_awgn, 'k--', 'LineWidth', 1.5, 'DisplayName', 'AWGN');

for mod_idx = 1:length(modulations)
    mod_type = modulations{mod_idx};
    if isstrprop(mod_type(1), 'digit'), f_name=['M', mod_type]; else, f_name=mod_type; end
    
    [sorted_corr, ~] = sort(RESULTS.(f_name).Corr);
    cdf_c = (1:length(sorted_corr))' / length(sorted_corr);
    
    bits = N_subcarriers * bits_per_symbol(mod_idx);
    lbl = sprintf('%s(%dbits)', mod_type, bits);
    
    plot(sorted_corr, cdf_c, 'LineWidth', 2, 'Color', colors{mod_idx}, 'DisplayName', lbl);
end
xlabel('Absolute Correlation Coefficient'); ylabel('CDF');
title(sprintf('Correlation CDF (Attack Period %.3f)', 1-T_ob));
legend('Location', 'southeast');
xlim([0 1]); ylim([0 1]);

fprintf('\nSimulation Complete. Generated BER and Correlation Plots.\n');
