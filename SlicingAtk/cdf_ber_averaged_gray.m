clear; close all; clc;

%% Parameters 
N_fft = 128;         % FFT size
N_subcarriers = 122; % Active subcarriers
SNR_dB = 30;         % SNR
T_ob = 5/8;          % Observation window
N_attack_trials = 10000; % Number of Packet trials
N_symbols_per_packet = 1; % Averaging block size

% Scaling for T_ob (energy loss in partial observation)
% Slicer treats signal as attenuated by T_ob (approx)
slicer_scaling_factor = T_ob; 

%% Modulations (Gray Mapping)
modulations = {'256QAM', '64QAM', '16QAM', '8PSK', 'QPSK'};
bits_per_symbol = [8, 6, 4, 3, 2];
colors = {'b', 'g', 'r', 'c', [0.5 0 0.5]}; 

%% Run simulations
fprintf('Running simulations (Gray Coding, Packet Avg): N_fft=%d, T_ob=%.3f, SNR=%d dB\n', N_fft, T_ob, SNR_dB);
fprintf('Packet Size: %d symbols\n', N_symbols_per_packet);

BER_all = cell(length(modulations), 1);

for mod_idx = 1:length(modulations)
    mod_type = modulations{mod_idx};
    fprintf('Simulating %s...\n', mod_type);
    
    bps = bits_per_symbol(mod_idx);
    M = 2^bps;
    
    ber_values = zeros(N_attack_trials, 1);
    
    % Prepare batch parameters
    % We process one packet at a time to calculate packet-averaged BER
    
    for trial = 1:N_attack_trials
        
        % 1. Generate Bits for the whole packet
        % Total bits needed = Subcarriers * bps * PacketSize
        n_total_bits = N_subcarriers * bps * N_symbols_per_packet;
        tx_bits = randi([0 1], n_total_bits, 1);
        
        % 2. Modulate (Gray Mapping + Unit Average Power)
        % Reshape to symbols: [PacketSize * N_subcarriers, 1]
        tx_bits_mat = reshape(tx_bits, bps, [])';
        tx_syms_idx = bi2de(tx_bits_mat, 'left-msb');
        
        if contains(mod_type, 'PSK')
             % PSK with Gray
             % Phase offset pi/4 for QPSK to match previous diag constellation if desired, 
             % but for BER vs SNR comparison standard starts at 0 is fine.
             % Previous QPSK was diagonals: pi/4. 
             % 8PSK was standard 0..7*pi/8.
             if strcmp(mod_type, 'QPSK')
                 tx_syms = pskmod(tx_syms_idx, M, pi/4, 'gray');
             else
                 tx_syms = pskmod(tx_syms_idx, M, 0, 'gray');
             end
        else
             % QAM with Gray and Unit Power
             tx_syms = qammod(tx_syms_idx, M, 'gray', 'UnitAveragePower', true);
        end
        
        % 3. Map to FFT & IFFT (Block Processing)
        % We have (N_subcarriers * PacketSize) symbols.
        % Reshape to [N_subcarriers, PacketSize]
        tx_syms_grid = reshape(tx_syms, N_subcarriers, N_symbols_per_packet);
        
        X_tx = zeros(N_fft, N_symbols_per_packet);
        % Map to 20MHz HE-LTF indices (approx)
        % Indices: 2:62 and 68:128
        X_tx(2:62, :) = tx_syms_grid(1:61, :);
        X_tx(68:128, :) = tx_syms_grid(62:122, :);
        
        % IFFT (Col-wise)
        x_tx = ifft(X_tx, N_fft);
        
        % 4. Channel (AWGN)
        % Add noise to full packet
        sig_pwr = mean(mean(abs(x_tx).^2));
        noise_pwr = sig_pwr / (10^(SNR_dB/10));
        noise = sqrt(noise_pwr/2) * (randn(size(x_tx)) + 1j*randn(size(x_tx)));
        x_rx = x_tx + noise;
        
        % 5. Observation (Truncation)
        n_obs = round(N_fft * T_ob);
        x_part = x_rx(1:n_obs, :);
        
        % 6. Zero Pad & FFT (Attacker)
        x_pad = [x_part; zeros(N_fft - n_obs, N_symbols_per_packet)];
        X_est = fft(x_pad, N_fft);
        
        % 7. Extract Data & Scale
        X_rec_data = zeros(N_subcarriers, N_symbols_per_packet);
        X_rec_data(1:61, :) = X_est(2:62, :);
        X_rec_data(62:122, :) = X_est(68:128, :);
        
        % Scale Compensation
        % Received energy is attenuated by T_ob approx. 
        % We scale UP to match Unit Power Constellation for Demod
        Y_rec = X_rec_data / slicer_scaling_factor;
        
        % 8. Demodulate (Gray Mapping)
        % Serialize for demod
        Y_rec_serial = Y_rec(:);
        
        if contains(mod_type, 'PSK')
             if strcmp(mod_type, 'QPSK')
                 rx_bits = pskdemod(Y_rec_serial, M, pi/4, 'gray', 'OutputType', 'bit');
             else
                 rx_bits = pskdemod(Y_rec_serial, M, 0, 'gray', 'OutputType', 'bit');
             end
        else
             rx_bits = qamdemod(Y_rec_serial, M, 'gray', 'UnitAveragePower', true, 'OutputType', 'bit');
        end
        
        % 9. Count Errors
        bit_errors = sum(rx_bits ~= tx_bits);
        ber_values(trial) = bit_errors / n_total_bits;
    end
    
    BER_all{mod_idx} = ber_values;
    fprintf('  Mean BER: %.4f, Median BER: %.4f\n', mean(ber_values), median(ber_values));
end

%% Plot combined CDF
figure('Name', 'Averaged BER CDF (Gray)');
hold on; grid on;
for mod_idx = 1:length(modulations)
    [sorted_ber, ~] = sort(BER_all{mod_idx});
    cdf_values = (1:length(sorted_ber))' / length(sorted_ber);
    
    total_bits_shown = N_subcarriers * bits_per_symbol(mod_idx);
    label = sprintf('%s(%dbits)', modulations{mod_idx}, total_bits_shown);
    
    plot(sorted_ber, cdf_values, 'LineWidth', 2.5, 'Color', colors{mod_idx}, 'DisplayName', label);
end

xlabel('Bit error rate', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('CDF', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('BER CDF (Gray Coded, Avg=100): T_{ob}=%.3f, SNR=%d dB', T_ob, SNR_dB), 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 10);
set(gca, 'FontSize', 11);
xlim([0 0.5]);
ylim([0 1]);
grid on;
fprintf('\nGray-Coded simulations complete!\n');
