using System;
using System.IO;
using System.Collections.Generic;

namespace TorGenHost
{
    class Program
    {

        static FileStream Fstream;// = new FileStream(; 
          private const int MAX_DNS_LABEL_SIZE = 63;
        private const double UINT_MAX = 4294967296.0;
        private static Random rnd = new Random();
        private static List<string> array_strig = new List<string>(); 
        static void Main(string[] args)
        {
            Console.WriteLine(".net and .com samples of names:");
            string dir = Directory.GetCurrentDirectory();
            string file_name = dir + "\\host_name4_25.txt";
            if (File.Exists(file_name))
            {
                //Fstream = new FileStream(file_name, FileMode.Truncate);
                Fstream = File.Open(file_name, FileMode.Truncate);
                Console.WriteLine("Файл:\"host_name4_25.txt-\" открыт");
            }
            else
            {
                Fstream = File.Open(file_name, FileMode.CreateNew);
                Console.WriteLine("Файл:\"host_nam4_25e.txt-\" создан и открыт");
                //File.WriteAllLines()
            }

            for (int i = 0; i < 1000000; i++)
            {
                //Console.WriteLine(crypto_random_hostname(8, 20, "www.", ".net"));
                
                string buf_str = crypto_random_hostname(4, 25, "www.", ".net")+'\r'+'\n';
                byte[] buf_byte = new byte[buf_str.Length];
                
                for (int n = 0; n < buf_str.Length; n++)
                {
                    buf_byte[n] =Convert.ToByte( buf_str[n]);
                }
                array_strig.Add(buf_str);
                Console.WriteLine(i.ToString()+"   :"+ buf_str);
                Fstream.Write(buf_byte, 0, buf_byte.Length);
                //File.Exists();
                //Console.WriteLine(crypto_random_hostname(8, 20, "www.", ".com"));
            }
            Fstream.Close();
            Console.Read();
        }
        static int crypto_rand_int(int max)
        {
            int val;
            double cutoff;

            cutoff = UINT_MAX - (UINT_MAX % max);
            while (true)
            {
                val = rnd.Next(429496729);
                if (val < cutoff)
                    return val % max;
            }
        }
        private static int crypto_rand_int_range(int min, int max)
        {
            return min + crypto_rand_int(max - min);
        }
        public static byte[] GenerateRandomBytes(int length)
        {
            byte[] randBytes;
            if (length >= 1)
            {
                randBytes = new byte[length];
            }
            else
            {
                randBytes = new byte[1];
            }
            // Create a new RNGCryptoServiceProvider.
            System.Security.Cryptography.RNGCryptoServiceProvider rand = new System.Security.Cryptography.RNGCryptoServiceProvider();
            // Fill the buffer with random bytes.
            rand.GetBytes(randBytes);
            // return the bytes.
            return randBytes;
        }
        private static string crypto_random_hostname(int min_rand_len, int max_rand_len, string prefix, string suffix)
        {
            string result;
            byte[] rand_bytes;
            int randlen;
            int rand_bytes_len;
            int resultlen;
            int prefixlen;

            if (max_rand_len > MAX_DNS_LABEL_SIZE)
                max_rand_len = MAX_DNS_LABEL_SIZE;
            if (min_rand_len > max_rand_len)
                min_rand_len = max_rand_len;

            randlen = crypto_rand_int_range(min_rand_len, max_rand_len + 1);
            prefixlen = prefix.Length;
            resultlen = prefixlen + suffix.Length + randlen + 16;
            rand_bytes_len = ((randlen * 5) + 7) / 8;
            if ((rand_bytes_len % 5) != 0)
            {
                rand_bytes_len += 5 - (rand_bytes_len % 5);
            }
            rand_bytes = GenerateRandomBytes(rand_bytes_len);
            result = Base32.Encode(rand_bytes);
            return prefix + result + suffix;
        }

    }
}
