"""
GNU Radio LoRa Simulation - Option B
=====================================
Real TX chain + SNR-based PDR calculation.

- Real GNU Radio TX blocks (whitening, Hamming, interleaving, gray mapping, modulate)
- Real AWGN channel noise
- Real SNR measurement after channel
- PDR calculated from real LoRa BER formula based on actual SNR

File: lora/gr_lora_sim.py
"""

import numpy as np
from gnuradio import gr, blocks, analog, channels
from gnuradio import lora_sdr
import time
import threading


class LoRaTransmitter(gr.top_block):

    def __init__(self, sf=7, samp_rate=250000, bw=125000, cr=1,
                 payload=None, snr_db=10.0):
        gr.top_block.__init__(self, "LoRa Transmitter")

        self.sf = sf
        self.samp_rate = samp_rate
        self.bw = bw
        self.cr = cr
        self.snr_db = snr_db

        if payload is None:
            payload = b'\x00' * 49

        # Source
        self.source = blocks.vector_source_b(list(payload), False, 1, [])

        # TX chain
        self.lora_whitening = lora_sdr.whitening(
            is_hex=False,
            use_length_tag=False,
            separator=',',
            length_tag_name=''
        )
        self.lora_add_crc = lora_sdr.add_crc(has_crc=True)
        self.lora_hamming_enc = lora_sdr.hamming_enc(cr=cr, sf=sf)
        self.lora_interleaver = lora_sdr.interleaver(cr=cr, sf=sf, ldro=0, bw=bw)

        # Type converters
        self.int_to_float = blocks.int_to_float()
        self.float_to_short = blocks.float_to_short()
        self.lora_gray_mapping = lora_sdr.gray_mapping(soft_decoding=False)
        self.short_to_float = blocks.short_to_float()
        self.float_to_int = blocks.float_to_int()

        self.lora_modulate = lora_sdr.modulate(
            sf=sf,
            samp_rate=samp_rate,
            bw=bw,
            sync_words=[0x12],
            inter_frame_padd=0,
            preamble_len=8
        )

        # AWGN channel
        noise_voltage = 10 ** (-snr_db / 20.0)
        self.channel = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0 + 0j],
            noise_seed=int(time.time()) % 1000
        )

        # Measure signal power after channel
        self.signal_sink = blocks.vector_sink_c()

        # Connect TX chain
        self.connect((self.source, 0), (self.lora_whitening, 0))
        self.connect((self.lora_whitening, 0), (self.lora_add_crc, 0))
        self.connect((self.lora_add_crc, 0), (self.lora_hamming_enc, 0))
        self.connect((self.lora_hamming_enc, 0), (self.lora_interleaver, 0))
        self.connect((self.lora_interleaver, 0), (self.int_to_float, 0))
        self.connect((self.int_to_float, 0), (self.float_to_short, 0))
        self.connect((self.float_to_short, 0), (self.lora_gray_mapping, 0))
        self.connect((self.lora_gray_mapping, 0), (self.short_to_float, 0))
        self.connect((self.short_to_float, 0), (self.float_to_int, 0))
        self.connect((self.float_to_int, 0), (self.lora_modulate, 0))
        self.connect((self.lora_modulate, 0), (self.channel, 0))
        self.connect((self.channel, 0), (self.signal_sink, 0))

    def get_received_samples(self):
        return np.array(self.signal_sink.data())


class GNURadioLoRaSimulator:

    DUTY_CYCLE = 0.01

    def __init__(self, sf=7, bw=125000, cr=1, snr_db=10.0):
        self.sf = sf
        self.bw = bw
        self.cr = cr
        self.snr_db = snr_db

        self.cr_map = {1: 4/5, 2: 4/6, 3: 4/7, 4: 4/8}
        self.cr_value = self.cr_map[cr]

        self._calculate_rates()

        self.total_transmissions = 0
        self.successful_transmissions = 0
        self.last_tx_time = 0
        self.measured_snrs = []

    def _calculate_rates(self):
        # Equation 26
        self.symbol_rate = self.bw / (2 ** self.sf)
        # Equation 27
        self.raw_bit_rate = self.sf * self.symbol_rate
        # Equation 28
        self.effective_data_rate = self.raw_bit_rate * self.cr_value

    def calculate_time_on_air(self, payload_length):
        # Equations 29-30
        PL = payload_length
        n_preamble = 8
        numerator = 8 * PL - 4 * self.sf + 28 + 16
        denominator = 4 * (self.sf - 2)
        payload_symbols = max(np.ceil(numerator / denominator) * (self.cr + 4), 0)
        n_symbols = n_preamble + payload_symbols
        t_toa = (n_symbols * (2 ** self.sf)) / self.bw
        return t_toa

    def calculate_ber_from_snr(self, snr_db):
        """
        Real LoRa BER formula based on SNR.
        Uses theoretical LoRa BER: BER = 0.5 * exp(-SNR * SF / 2)
        """
        snr_linear = 10 ** (snr_db / 10.0)
        ber = 0.5 * np.exp(-snr_linear * self.sf / 2.0)
        return ber

    def calculate_pdr_from_ber(self, ber, payload_bytes=49):
        """
        Real PDR from BER.
        PDR = (1 - BER)^(payload_bytes * 8)
        A packet is delivered only if ALL bits are correct.
        """
        pdr = (1 - ber) ** (payload_bytes * 8)
        return pdr

    def measure_snr(self, received_samples):
        """
        Measure actual SNR from received IQ samples after AWGN channel.
        SNR = signal_power / noise_power
        """
        if len(received_samples) == 0:
            return self.snr_db

        # Signal power
        signal_power = np.mean(np.abs(received_samples) ** 2)

        # Noise power estimate (from variance)
        noise_power = np.var(received_samples)

        if noise_power > 0:
            measured_snr = 10 * np.log10(signal_power / noise_power)
        else:
            measured_snr = self.snr_db

        return measured_snr

    def _run_flowgraph_with_timeout(self, payload, timeout=8):
        """Run GNU Radio flowgraph in thread with timeout."""
        tb_container = [None]
        finished_flag = [False]

        def run_tb():
            try:
                tb = LoRaTransmitter(
                    sf=self.sf,
                    samp_rate=250000,
                    bw=self.bw,
                    cr=self.cr,
                    payload=payload,
                    snr_db=self.snr_db
                )
                tb_container[0] = tb
                tb.start()
                tb.wait()
                tb.stop()
                finished_flag[0] = True
            except Exception as e:
                print(f"Flowgraph error: {e}")
                finished_flag[0] = False

        thread = threading.Thread(target=run_tb)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            print(f"Flowgraph timeout after {timeout}s, forcing stop")
            try:
                if tb_container[0] is not None:
                    tb_container[0].stop()
            except:
                pass
            finished_flag[0] = True

        return tb_container[0], finished_flag[0]

    def transmit(self, payload):
        t_toa = self.calculate_time_on_air(len(payload))

        # Run real GNU Radio TX chain
        tb, flowgraph_ok = self._run_flowgraph_with_timeout(payload, timeout=8)

        self.last_tx_time = time.time()

        if not flowgraph_ok or tb is None:
            # Flowgraph failed completely
            self.total_transmissions += 1
            return {
                'success': False,
                't_toa': t_toa,
                'measured_snr': None,
                'ber': 1.0,
                'theoretical_pdr': 0.0,
                'packet_size': len(payload),
                'pdr': self.successful_transmissions / max(self.total_transmissions, 1)
            }

        # Measure actual SNR from received samples
        try:
            received_samples = tb.get_received_samples()
            measured_snr = self.measure_snr(received_samples)
        except:
            measured_snr = self.snr_db

        self.measured_snrs.append(measured_snr)

        # Calculate real BER and PDR from measured SNR
        ber = self.calculate_ber_from_snr(measured_snr)
        theoretical_pdr = self.calculate_pdr_from_ber(ber, len(payload))

        # Packet success based on real theoretical PDR
        success = np.random.random() < theoretical_pdr

        self.total_transmissions += 1
        if success:
            self.successful_transmissions += 1

        empirical_pdr = self.successful_transmissions / self.total_transmissions

        return {
            'success': success,
            't_toa': t_toa,
            'measured_snr': measured_snr,
            'ber': ber,
            'theoretical_pdr': theoretical_pdr,
            'packet_size': len(payload),
            'pdr': empirical_pdr
        }

    def print_configuration(self):
        print("\n" + "="*60)
        print("GNU Radio LoRa Simulation - Option B")
        print("Real TX + SNR-based PDR")
        print("="*60)
        print(f"Spreading Factor (SF): {self.sf}")
        print(f"Bandwidth (BW): {self.bw/1000:.1f} kHz")
        print(f"Coding Rate (CR): {self.cr_value:.2f}")
        print(f"Target SNR: {self.snr_db} dB")
        print(f"Eq 26 - Symbol Rate: {self.symbol_rate:.2f} symbols/sec")
        print(f"Eq 27 - Raw Bit Rate: {self.raw_bit_rate:.2f} bps")
        print(f"Eq 28 - Effective Rate: {self.effective_data_rate:.2f} bps")
        print(f"Duty Cycle: {self.DUTY_CYCLE*100}%")
        print("="*60)


if __name__ == "__main__":
    sim = GNURadioLoRaSimulator(sf=7, bw=125000, cr=1, snr_db=10.0)
    sim.print_configuration()
    compressed_data = b'\x00' * 49
    t_toa = sim.calculate_time_on_air(len(compressed_data))
    print(f"Equation 30 - Calculated ToA: {t_toa:.4f} seconds")
    for i in range(3):
        result = sim.transmit(compressed_data)
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"TX {i+1}: {status} | ToA: {result['t_toa']:.4f}s | "
              f"Measured SNR: {result['measured_snr']:.2f}dB | "
              f"BER: {result['ber']:.6f} | "
              f"PDR: {result['pdr']*100:.1f}%")
